# Rank-routing implementation hold

The source-only draft is preserved for repair, not admitted for native execution. Root draft checks and the historical source suite passed, but independent engineering review found three source blockers: retained split-MSE graph ownership, missing live authentication of the derived regression callable/private globals, and missing independent full-B64 versus micro16 correspondence.

Fix only these demonstrated gaps, retain the original scalar objective, routing mechanism, schedules, caps, complete source/native exits and historical byte inverse. First reproduce each failure with a narrow source falsifier, then repair and verify. A new final source descriptor and fresh CPU/mechanics qualification are required before TRAIN. Earlier passing stand-ins establish no native parity or quality result.

Engineering reviewer result (original consultation 04b051cfaa0142a7):

+ What this change does: preserves the forward objective while routing candidate encoder gradients through SmoothAP only. Both supplied file hashes match.
 
 **Verdict: source NO-GO; fix these three issues before native qualification.**
 
 1. **[P1] The CPU split witness retains the graph it checks for destruction.**  
    [Trainer:1170](/home/rb/worktrees/sfora-connected-rank-routing-20261009/scripts/train_siglip2_rank_routed_mlp.py:1170) assigns the returned MSE tensor to `_`. That local remains alive through the weak-reference check at line 1185, retaining the query-side autograd graph and its A/C leaves. The [test stand-in:434](/home/rb/worktrees/sfora-connected-rank-routing-20261009/scripts/test_siglip2_rank_routed_mlp.py:434) returns `None` for MSE, masking this ownership problem.
 
    **Minimal repair:** name it `split_mse` and delete it before checking temporary lifetimes. **Smallest falsifier:** make the stand-in return an MSE object retaining `raw`; the current lifetime check should reject it and the repaired version should pass. The retained local is source-supported; the expected native failure was not executed.
 
 2. **[P2] Construction authentication does not protect the live derived regression callable.**  
    [_regression_runtime:449](/home/rb/worktrees/sfora-connected-rank-routing-20261009/scripts/train_siglip2_rank_routed_mlp.py:449) authenticates the original `loss_terms` and derived AST, then returns a mutable namespace. [Update:1072](/home/rb/worktrees/sfora-connected-rank-routing-20261009/scripts/train_siglip2_rank_routed_mlp.py:1072) calls its current `regression_terms` without checking the admitted callable, code, private globals, or routing-context binding. Copied helper functions such as `loss_denominators` also remain mutable.
 
    A replacement returning the genuine regression scalar’s `.detach()` after step17 preserves scalar parity while removing A/C regression gradients; neither the step1 routing witness nor first17 replay covers that change.
 
    **Minimal repair:** bind and check the admitted runtime/callable/code and required private-global dependencies at use, following the existing authentication pattern. **Smallest falsifiers:** replace the runtime/callable, alter its code, and replace a private-global helper after construction; each must fail before objective evaluation. Existing tests primarily mutate the original function before construction.
 
 3. **[P2] The required independent full/micro witness is missing.**  
    [route_view:1214](/home/rb/worktrees/sfora-connected-rank-routing-20261009/scripts/train_siglip2_rank_routed_mlp.py:1214) compares routed and original **sums of the same four microbatch computations**. Its `full_A_C_total_max_abs` field establishes accumulated routing correspondence, but no independent B64-versus-micro16 comparison occurs. [cpu_run:1885](/home/rb/worktrees/sfora-connected-rank-routing-20261009/scripts/train_siglip2_rank_routed_mlp.py:1885) adds no such computation, despite the explicit [plan requirement:13](/home/rb/worktrees/sfora-connected-rank-routing-20261009/docs/connected_rank_routing_plan_2026-10-09.md:13).
 
    **Minimal repair:** add the planned independent first-B64 objective comparison against micro16 accumulation for both views, with receipt enforcement. **Smallest falsifier:** perturb one micro contribution equally in the routed and original accumulators; their current mutual comparison remains satisfied, while an independent full reference must reject it. Keep the existing caps.
 
 The normal source path otherwise uses the pinned regression expression, detached features for regression, and the original ranking objective. I found no additional change-induced issue in schedule128, clipping/scaler, frozen444 checks, first17 replay, 8+9 restoration, or cost accounting.
 
 After repair and source refreezing, require CPU600, paired mechanics1200, then fresh061 control/candidate TRAIN3000; retain complete payload/lifetime/uncached-exit checks, existing resource limits, and both core and normal-exit service ratios ≤1.50. FIRST remains the gate to069.
 
 Not checked: no edits, tests, native execution, jobs, or children; the supplied source-pass log was inspected only. Native correctness, performance, and quality remain unqualified.
