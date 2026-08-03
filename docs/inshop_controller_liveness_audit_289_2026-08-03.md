# In-Shop controller liveness audit 289

Date: 2026-08-03.

After every required SOP artifact and independent verification had been
published, the serialized In-Shop controller exited without launching the
registered reference. It left no report, checkpoint, partial artifact, or GPU
process. The scientific queue was unchanged, but the failure was silent because
the original launch did not preserve a dedicated controller log.

The corrected In-Shop reference was relaunched only after rechecking all required
SOP evidence, the exact recipe digest, and absence of stale targets. Its final
export controller contained a second stale dependency: it treated the obsolete
`run_inshop_after_sop_repair.sh` wrapper as producer liveness. The controller now
waits for the exact `sfora image-end-to-end` command containing the registered
output stem, requires the report and checkpoint after that producer exits, and
logs the independent export explicitly.

This is an orchestration repair, not a method or result. It matters because a
stale wrapper name could either terminate verification early or keep it waiting
for the wrong process, reproducing the same class of liveness bug found in the SOP
chain.
