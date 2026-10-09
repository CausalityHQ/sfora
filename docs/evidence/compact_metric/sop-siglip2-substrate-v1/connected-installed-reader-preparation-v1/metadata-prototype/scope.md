# Bounded metadata candidate, root scratch preparation

Implements the runtime-owned raw metadata read required by review C3 without
using the admission module's private descriptor API. Eight local stdlib cases
pass under timeout15/AS1GiB: unchanged bytes, wrong pin, same-size mutation with
restored mtime, symlink, hardlink, FIFO, oversized sparse file, and simultaneous
read/close failure preserving the exact primary and secondary note. No model,
image, tensor, native library or production runtime execution.

This is not integrated code and is not a separately approved public API.
It requires an independently pinned path/SHA from the admitted artifact.
Canonical path checks, a regular single-link fd and size/hash/current inode
checks bind captured bytes; they do not create an atomic parent-directory or
multi-file snapshot or prevent later mutation. It grants no native permission.
Full artifact admission and later fresh exit/use checks remain mandatory.

The installed-reader child remains clean and on implementation HOLD; root made
no overlapping edits to its owned runtime/authority/tests. The candidate is
available for the eventual finite reviewed loader implementation, not proof
of that loader's composition, installed-wheel admission or native parity.
