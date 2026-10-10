**Verdict: fix findings 1–4 before native launch.** Reviewed source guard `05845336` and relocation helper `ae7e2214`. The counterexamples below are derived from source; I did not execute them.

1. **[P1] Missing filesystem modules can fall through to an unapproved loader.**  
   [find_spec, line 192](/home/rb/worktrees/sfora-positive-causality/scripts/connected_installed_source_guard.py:192) returns `None` whenever `PathFinder` misses. That permits every later meta-path finder, not just the intended builtin/frozen finders.

   Executable falsifier using the existing test fixture `t`:
   ```python
   name = t.name + ".virtual"
   class Virtual:
       def find_spec(self, fullname, path=None, target=None):
           if fullname == name:
               return importlib.util.spec_from_loader(fullname, self)
       def create_module(self, spec):
           return None
       def exec_module(self, module):
           module.value = "unapproved"

   sys.meta_path.append(Virtual())
   with t.gate() as gate:
       t.load(gate)
       assert importlib.import_module(name).value == "unapproved"
       gate.check()
   ```
   No declared source supplies this module, yet execution and exit checks succeed.

   **Smallest repair:** handle genuine builtin/frozen resolution explicitly and reject other unresolved imports, or authenticate the exact permitted finder chain. The falsifier must reject without executing `Virtual.exec_module`; ordinary missing imports must retain `ImportError` semantics.

2. **[P1] Added registry entries escape the ownership check.**  
   [check, line 238](/home/rb/worktrees/sfora-positive-causality/scripts/connected_installed_source_guard.py:238) checks only `self.loaded`. The protected-prefix registry scan occurs only on entry.

   After loading the fixture:
   ```python
   name = t.name + ".rogue"
   foreign = types.ModuleType(name)
   sys.modules[name] = foreign
   assert importlib.import_module(name) is foreign
   gate.check()  # No rejection
   ```
   This also applies beneath declared namespaces. Existing tests replace an already tracked entry; they do not add an untracked one.

   This matters to the actual source: the recorded Transformers initializer creates fileless alias modules at decoded lines 811–858 of [lazy-source-observation.json](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-installed-source-guard-v1/lazy-source-observation.json:1). Those aliases never enter `self.loaded`, so their subsequent replacement is invisible.

   **Smallest repair:** reconcile protected registry entries against explicit ownership at checks and exit. Legitimate generated aliases need finite provenance and identity checks; a package-prefix exception would leave the hole open. Test both an added entry and replacement of a legitimate alias.

3. **[P1] Lazy-class globals can change while authentication passes.**  
   [replacement_actor, line 163](/home/rb/worktrees/sfora-positive-causality/scripts/connected_installed_source_guard.py:163) checks that each method retains the same globals dictionary. It does not check the bindings inside that dictionary.

   With the existing lazy fixture loaded:
   ```python
   p = sys.modules[provider]
   p.chain = lambda _: ("foreign",)
   gate.check()  # Passes
   x = p._LazyModule("x", module.__file__, {"child": ["value"]},
                     module_spec=module.__spec__)
   assert x.__all__ == ["child", "foreign"]
   ```
   Class identity, method code and source bytes remain unchanged, while constructor behavior changes. The recorded full class has a stronger consequence: `_get_module` calls global `importlib.import_module` at decoded line 2494. Rebinding that provider-global `importlib` can return a foreign object without consulting the source finder.

   **Smallest repair:** retain and verify the specific provider-global bindings used by the admitted class. Add a falsifier that changes `chain` in the current fixture and requires rejection. The current code-tampering test does not cover this case.

4. **[P2] Profile activation can fail before restoration is protected.**  
   [exec_module, line 54](/home/rb/worktrees/sfora-positive-causality/scripts/connected_installed_source_guard.py:54) calls `sys.setprofile(capture)` before entering its restoration `try/finally`. The callback can run on that call’s `c_return` event and immediately raise through `replacement_actor`.

   Concrete falsifier: load the lazy fixture, install a harmless existing profiler, change the constructor code as the current tampering test does, then import the still-unloaded declared child:
   ```python
   prior = lambda *args: None
   sys.setprofile(prior)
   cls.__init__.__code__ = saved.replace(co_filename="/foreign.py")
   try:
       importlib.import_module(t.name + ".child")
   except ValueError:
       pass
   assert sys.getprofile() is prior
   ```
   The callback rejection disables profiling before the restoration block is entered, so the final assertion fails. This contradicts the contract’s restoration-on-failure promise.

   **Smallest repair:** include profile activation inside the protected region and restore the previous hook on activation failure. Preserve the original rejection. The existing test mutates code after import and calls `gate.check()`, so it never exercises this path.

I found no additional concrete overgrant in the relocation helper’s reviewed H-table/S boundary. [Exact-image validation](/home/rb/worktrees/sfora-installed-native-relocation-20261010/scripts/connected_installed_native_relocation.py:201) and the [unsplit-S rejection test](/home/rb/worktrees/sfora-installed-native-relocation-20261010/scripts/test_connected_installed_native_relocation.py:461) support its stated data-only scope. I have not repeated the known R1–R6 integration blockers.

The source and test hashes match the [11-test verification receipt](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-installed-source-guard-v1/verification.json:5). Its constructor-only scope is an evidence limit, not proof of full Transformers compatibility. The 7,183-source pool and 56 unselected extensions remain ungranted.

Not checked: runtime falsifiers, tests, ML/native execution or performance; no files changed.  
Risk: root must reconcile these findings before launch; full native compatibility and joint quality/speed remain unestablished.
