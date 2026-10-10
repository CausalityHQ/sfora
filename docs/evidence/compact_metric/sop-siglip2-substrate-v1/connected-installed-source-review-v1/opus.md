I'd hold the native launch on one source defect, plus two protocol gaps that would quietly turn the guard off. The relocation helper has no blocker within its data-only scope.

I only read files and checked hashes: no edits, tests, ML imports, SSH, native code, jobs, children or consults. The pins in `verification.json` match the committed files (3eda5d81…, 4dead6dc…, log f66de6cd…), and the test count of 11 matches.

## Source guard (`scripts/connected_installed_source_guard.py`)

### Defect to fix before native launch

**F1. A rejection can be swallowed and leave no evidence at exit, so the contract's word "reject" is not true.**
- **Code:** the denials at L196–197 and L204–205 raise `ImportError`. The `exec_module` failure path at L83–86 removes the module from `loaded`. `check()` at L238–243 only looks at `loaded` and `consumed`, so nothing records that a denial happened.
- **Causal example:** a managed `pkg/__init__.py` contains `try: from . import _ext` / `except ImportError: _ext = None`. `_ext.cpython-313-aarch64-linux-gnu.so` exists but is one of the 56 ungranted extensions. `PathFinder` resolves the `.so`, L204 raises, the module catches it and runs its fallback, and `close()` passes.
- **Real instances in the recorded Transformers source:**
  - `import_utils.py:1218-1224`: `is_vision_available` catches `ImportError` from `import PIL.Image`, is `@lru_cache`d, and runs inside the `_LazyModule.__init__` backend loop (the frozenset branch that `__init__.py:800-809` actually uses). One denial anywhere in PIL's import chain marks "vision" unavailable for the rest of the process.
  - `import_utils.py:1442-1449` and `1463-1469` catch `Exception`, so they also swallow guard `ValueError`s (L206, and `check_module` failures).
- **Not caught downstream either:** `collect()` (`connected_control_native_authority.py:259-263`) accepts a subset of H. A missing native is not rejected there.
- **Already covered:** source mutation is still caught, because `consumed` keeps the path (L187). The gap is specifically denials and identity/route rejections.
- **Smallest repair:** add `self.rejected = []`. Append at every guard raise in `find_spec` and in the `except` at L83, before re-raising. Make `check()` require `not self.rejected`.
- **Falsifier:** the fixture above must make `close()` raise. Add a second variant that catches `except Exception` around an L206 name/path mismatch.

### Protocol gaps that defeat the guard

**F2. Imports that happen after `close()` are not guarded.**
- `close()` (L252–258) removes the finder. Several imports happen later:
  - `transformers` is a `_LazyModule`, so model modules import on first attribute access.
  - The aliases at `__init__.py:811-858` import their targets lazily.
  - PIL imports its plugins inside `Image.open`.
- After `close()`, those imports go through the normal `PathFinder`. A valid `__pycache__` `.pyc` is accepted and `ExtensionFileLoader` loads any `.so`, so the bytecode denial no longer applies.
- **Falsifier:** after `gate.close()`, `import_module(name + '.late')` with a valid poisoned cache returns the poisoned value.
- **Repair:** the guard's scope must cover every serving request up to the final uncached exit. The root's protocol has to pin that; code alone can't.

**F3. The import machinery around the guard is not pinned.**
- L239 only checks that the finder is registered once.
- **Falsifier:** after `__enter__`, run `sys.meta_path.insert(0, importlib.machinery.PathFinder)`. Later managed imports load cached `.pyc` files and `check()` still passes.
- An appended finder can also serve names that L192–193 hands back as `None`, for example PEP 660 editable finders or the `_distutils_hack` shim.
- **Repair:**
  - Snapshot `tuple(sys.meta_path)` at enter, after the native finder is composed in front, and require it to be identical in `check()`.
  - Also scan `sys.modules`: any module under a protected top-level name whose file origin is inside an installed root must be in `loaded`. Exclude the file-less alias modules from `__init__.py:815-819`.

### Should-fix

**F4. The profile hook is much more expensive than it needs to be.** This is a compatibility and speed issue, not a correctness one.
- `capture` (L46–53) calls `replacement_actor` (the full class and function identity sweep at L157–165) on every call, C-call and return event, before it checks the event type or code.
- The route prefix (L147–151) installs the hook for every `transformers.*` module, including non-package modules that `check_module` L236 can never accept as replacements. Nested imports chain `previous`, so the cost multiplies with import depth.
- In the actual source the actor is already set up by `__init__.py:57`, so the full sweep runs during:
  - `define_import_structure(models)` (L800), which opens and parses every model file (`import_utils.py:2770`);
  - the backend probes inside the constructor;
  - the first-access lazy load of modeling modules at serving time, which adds to first-request latency.
- **Repair:**
  - Test `event == 'return' and frame.f_back is not None and frame.f_back.f_code is code` before calling `replacement_actor`.
  - Install the hook only when the module's file is `__init__.py`.
  - The binding checks after execution (L65–69) and at exit are unchanged.

**F5. The stdlib root is granted as a whole directory prefix (L201–203).**
- Any `sys.path` entry under a stdlib root falls through unauthenticated, including `.so` and `.pyc` files.
- On python-build-standalone, uv and python.org layouts, `<prefix>/lib/python3.13/site-packages` sits under the stdlib root. L108 only checks overlap with the installed roots.
- I infer the DGX base interpreter is statically linked like python-build-standalone, because the 13 anchors include no libpython, libcrypto or lib-dynload files. I have not verified this.
- **Falsifier:** with `stdlib_roots=(T,)` and `sys.path=[T/'site-packages']`, an `evil.py` there imports through the guard.
- **Repair:** pin `sys.path` exactly (installed roots plus the declared stdlib entries) at enter and in `check()`.

**F6. Optimization level differs from the normal loader.** L36 compiles with `optimize=0`, while the normal loader uses `-1` (follow `sys.flags.optimize`). The two only match if the interpreter runs without `-O`.
- **Repair:** use `optimize=-1`, or require `sys.flags.optimize == 0` at enter.

**F7. One module name can map to two files.** L124–125 doesn't require that. For example, `pkg.py` and `pkg/__init__.py` in the same root, or the same name in two roots, lets `sys.path` order decide which runs.
- **Repair:** require one row per module name.

**F8. Wording only.** L50–53 accepts any `__init__` return whose caller is the module code. That includes `actor.__init__(foreign_obj, …)` re-initialising an object built elsewhere. Only authenticated code could do this, so it isn't exploitable, but "constructor-return instance" overstates what is checked.

### Limits of the source-only tests (not defects)
- The "old_root" test (L87–96) puts `foreign` inside the installed root and mocks `PathFinder`. It goes through the same L204 branch, but real resolution outside the roots is never exercised.
- The lazy-module tests use the non-frozenset constructor branch. These parts of the real `__init__.py` are not covered:
  - backend probes and nested imports under the profile hook;
  - the alias registry entries (L811–858);
  - the `rglob` at L834.
- The import structure is built from `open()` reads the guard doesn't see (`import_utils.py:2770`), which is left to the outer I/O audit. The remaining risk is a name being routed to a different, still-granted module.
- `_is_package_available` (L52) calls `find_spec`. For a package that is installed but not granted, the guard raises `ImportError` rather than returning `None`, and the constructor's `except (ModuleNotFoundError, RuntimeError)` doesn't catch it, so the failure is loud. The consequence is that the file table must include the `__init__.py` of every installed package that any probe touches.

## Relocation helper (`connected_installed_native_relocation.py`)

No blocker. The package equality in the inverse matches `collect()`, which compares packages with `==` at L251. As you asked, I did not repeat the R1–R6 reads or the seam design.

- **RL1 (low): the hardlink check is too narrow.**
  - The alias check (L259–268) only compares against the original file with the same label, and the duplicate-inode check (L282) only covers installed files.
  - An installed file hardlinked to a different old file with identical bytes therefore passes. Many empty `__init__.py` files share the same digest, so this is reachable.
  - **Repair:** compare against the set of every selected original's `(dev, ino)` via `lstat`, without reading bytes. Don't use `st_nlink == 1`: uv's hardlink install mode makes link counts above 1 legitimate.
- **RL2 (docstring):** L63–64 says passing whole H+S is rejected. That only holds for S paths outside the table. An S path equal to one of the 13 anchors maps to itself and is accepted. That is harmless, but the docstring should say so, and a test could cover S overlapping the anchors.
- **RL3 (latent nit):** L325 uses `str.splitlines()`, which also splits on `\x0c`, `\x85` and `\u2028`, while `ast` line numbers count only `\n`, `\r\n` and `\r`. None of the current seam files contain those characters (I checked), so nothing is affected today. `split('\n')` is the safer choice.

Across both components: once F1 is fixed, any import the installed run makes beyond the original H set (1754 selected files) fails loudly instead of degrading silently. I make no claim about full-framework compatibility, profile overhead on the actual hardware, native parity, or the combined quality and speed goal, which is still unmet.
