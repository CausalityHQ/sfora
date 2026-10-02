Recommend one prospective evaluator change: **retain a populated, read-only mapping of the authenticated encoder checkpoint `fresh_vision.pt` for the measured invocation.** This is a falsifiable performance hypothesis; no speedup is established.

The audited source matches the supplied commit. HEAD has advanced to `91820472`, which adds the scorev2 timeout evidence.

The repeated path is:

| Boundary | Unchanged authentication |
|---|---|
| `load_head → fitter.reload → reconstruct → prepare_native` | Full fitted-checkpoint SHA, warm/encoder/canonical SHAs, fresh reconstruction and complete typed-state checks |
| `reload` and `load_head` completion | Each invokes `fitter.integrity` |
| `head_values → raw_features` | Integrity before and after inference; each hashes warm, encoder, canonical and solver files |
| Exit | Fresh complete guard-union SHA reads, original source/origin/composition predicates and closure checks |

That gives **21 full encoder-file hashes during CPU qualification and 29 during scoring**, excluding admission and exit. Its vision payload alone is **1,711,552,256 bytes**: approximately **35.94 GB / 49.64 GB** traversed respectively. These are logical byte counts, not measured removable disk traffic. CPUv6’s filesystem-input counter corresponds to **168,252,489,728 bytes** across the timed command.

The original shared [`bound_file`](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_substrate_adaptation.py:102), fitter and evaluators all issue consumed-range `DONTNEED`. Existing `FlatAdmission` deduplicates admission reads; fitter integrity uses its unconditional hash routine. Leave both behaviors intact.

1. **Integrate only in the prospective evaluator.** In [`native_start`](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_prototype_residual.py:653), after complete admission/prerequisite and cgroup checks, immediately before `fitter.prepare_native`, obtain the checkpoint through `trainer.owned_encoder(selected)`. Perform a fresh unchanged `bound_file` verification, open it read-only, create `mmap.ACCESS_READ`, and touch one byte per native page. Keep that mapping in evaluator-owned context through exit authentication and receipt checks; close it in `finally`. Inference continues using the original files and loaders.

   Preserve fitter source-v3, its execution hash, accepted endpoints, original modules/globals and all four independent reloads. Freeze a new evaluator closure, authority and launcher hashes.

2. **Retain exactly that one file.** Do **not** map warm or fitted checkpoints: [`CheckpointPages`](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_substrate_adaptation.py:1240) requires exactly one matching VMA, so an extra mapping would fail an unchanged predicate. The encoder’s [`encoder_metadata`](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_quadratic_readout.py:644) path has no such callback.

   Freeze the encoder’s actual serialized length **B** before launch; local evidence provides tensor bytes, not exact file length. Declare a maximum **2 GiB retention allocation**, with no expansion. Conservative memory allowance is existing score peak + B + page tables: **≤3.714 GB plus roughly 4 MiB page tables**, below the unchanged 8 GiB limit. This is budgeting, not qualification evidence.

3. **Smallest admissible native falsifier: one cold CPU300 qualification.** First add one small stdlib fixture check proving retention still rejects a same-size file mutation on the next full SHA read. Then root runs the complete newly frozen CPU invocation once, preserving cold-cache setup, both locks, original interpreter, CUDA hiding, 8 GiB/noSwap, zero events, launcher pre/post SHA lists and both footers.

   Require normal terminal acceptance, exact CPUv6 head/TRAIN witnesses, and reductions in both whole-service time and whole-command filesystem inputs against **299.405s / 328,618,144 blocks**. A pass permits one separately cold **score300** invocation with every existing source, parity, persisted-wire and scientific check.

The mechanism is plausible because Linux’s advisory invalidation excludes mapped pages, but pressure can still reclaim them; unchanged advice also retains syscall overhead. [Linux implementation](https://github.com/torvalds/linux/blob/master/mm/truncate.c)

**Stop rule:** any size-bound, identity, reload, parity, origin, terminal or resource failure—or no CPU time/I/O improvement—closes this intervention before scoring. A score timeout closes it without retries or hot-set expansion. Existing fits and CPUv6 remain valid; partial scorev2 wires remain unqualified. Production quality and public speed still require their existing checks.

No files changed, Torch imported, or native jobs started.

Root metadata verification: authenticated DGX stat gives fresh_vision.pt exactly 1,711,945,083 bytes. Retain only that file; fixed maximum 2 GiB, unchanged whole-unit 8 GiB/noSwap. Native CPU must improve both service time and filesystem-input blocks relative to CPUv6 299.405s / 328,618,144 blocks and preserve exact complete head/TRAIN facts before one cold score300. No retry or hot-set expansion after failure.
