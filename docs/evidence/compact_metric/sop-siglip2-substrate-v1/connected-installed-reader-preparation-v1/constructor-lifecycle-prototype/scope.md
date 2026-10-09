# Artifact-only constructor cleanup candidate

This is a prospective replacement for construct_encoder, not a production edit.
The numerical try-body and return AST are identical to the current constructor.
Without the private _serving context key, the original cleanup AST is retained,
including its original error precedence. With that key, collection and mapping
checks are attempted separately. The directly caught body exception remains
primary; secondary failures are notes. Without a body exception, the first
cleanup failure is raised and later failures are notes. No failed mapping is
converted to success, no alias is cleared, and no memory/resource cap changes.

The source-only check runs the actual candidate function with fake Torch,
processor, model, page and numerical dependencies and tiny opaque file bytes.
Sixteen combinations cover route/body failure/collection failure/mapping
failure. --original runs the same cases against the actual original function
and fails because collection failure skips the mapping check. A first fixture
run failed on missing packages in the test context; only that fixture was
repaired. No production failure or success is inferred from these stand-ins.

Errors before the constructor's existing try remain loader-owned. Finished
frames can retain native mappings; real alias disposal and foreign-owner
rejection still belong to the full loader lifecycle. Native model reload,
base448/overlay4 numeric parity, CUDA/resource fit and full uncached exit remain
unrun. The private context key selects cleanup behavior, not native permission.
