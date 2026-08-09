# Pass201 PA source-v2 closeout — 2026-08-09

## Verdict

Pass201 PA source-v2 is **BLOCKED and closed without a scientific result**.
No authorization manifest, source checkpoint, report, receipt, metric, Pass201
candidate value, or GPU training process was produced.

## Deciding failure

The first r4 `freeze-authority` process started from the clean detached checkout
`/home/riomus/sfora-pass201-pa-source-v2-r4` at source commit
`eea86dd1660f54c1c18a3aac35e7ea4bc6db4c4e`. It failed while binding the Python
package environment. The controller called the bound interpreter as:

```text
/home/riomus/group-learning/.venv/bin/python -m pip freeze --all
```

The bound interpreter is Python 3.13.9 and has no importable `pip` module. A
read-only reproduction after the stopped attempt returned:

```text
/home/riomus/group-learning/.venv/bin/python: No module named pip
exit=1
```

The failure occurred before either temporary authority output was published.
The canonical manifest and private run directory remained absent, the checkout
remained clean, and `nvidia-smi` reported no compute process.

The r4 amendment states that any failure after the first `freeze-authority`
process starts permanently stops this source attempt. Therefore there is no r5,
no retry with a modified environment, and no reinterpretation of r4.

## Root cause and impact boundary

The source-v2 controller treated `python -m pip` as part of the bound Python
interface without proving that the selected interpreter contained `pip`. The
tests used synthetic or local environments where that assumption held. This is
an environment-binding bug in a prospective source-generation path, not a model,
loss, dataset, or metric bug.

It invalidates no completed benchmark result or historical method rejection:
Pass201 source-v2 never produced a source artifact or candidate statistic. The
existing bug-impact audit remains valid. Pass204's mechanism attribution was
already invalid for its independent missing-control and coefficient-confound
reasons; this failure adds no evidence for or against CIS.

If a future source protocol is designed, package authority must use an interface
proved present before the non-repeatable boundary, such as a separately bound
package-manager executable or a strict `importlib.metadata` distribution export.
That would be a new protocol and attempt, not a repair or continuation of
Pass201 source-v2.
