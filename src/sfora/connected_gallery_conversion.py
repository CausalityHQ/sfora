"""Select opaque gallery wire rows from independently accepted producer bytes.

This returns a source fragment, not an admitted serving artifact. Ownership
evidence stays opaque; actual-wire, payload and native qualification are separate.
The ID list is a NEW consumer projection of exact producer relative_path strings.
"""

import hashlib

from .connected_gallery_provenance import (
    _canonical,
    _count,
    _list,
    _object,
    _require,
    _sha,
    _string,
    bind_gallery_provenance,
)


def extract_gallery_members(
    receipt: bytes,
    bundle: bytes,
    ownership_audit: bytes,
    combined_wire: bytes,
    *,
    trusted_receipt_sha256: str,
    trusted_bundle_sha256: str,
    trusted_ownership_audit_sha256: str,
) -> dict[str, bytes]:
    """Authenticate originals before parsing, then preserve their exact association.

    All inputs must be builtin immutable bytes. Invalid pins, producer metadata,
    wire lengths/hashes or IDs raise ValueError. No original path is opened.
    New output hashes never replace historical producer observations.
    """
    _require(
        all(type(raw) is bytes for raw in (receipt, bundle, ownership_audit, combined_wire)),
        "immutable bytes required",
    )
    for raw, pin, name in (
        (receipt, trusted_receipt_sha256, "receipt"),
        (bundle, trusted_bundle_sha256, "bundle"),
        (ownership_audit, trusted_ownership_audit_sha256, "ownership audit"),
    ):
        _require(hashlib.sha256(raw).hexdigest() == _sha(pin), f"trusted {name} SHA differs")
    producer = bind_gallery_provenance(
        receipt,
        bundle,
        trusted_receipt_sha256=trusted_receipt_sha256,
        trusted_bundle_sha256=trusted_bundle_sha256,
    )
    combined = _object(producer["combined_wire"])
    _require(len(combined_wire) == _count(combined["count"]) * 130, "combined wire length differs")
    _require(
        hashlib.sha256(combined_wire).hexdigest()
        == _object(combined["files"])["control-179061.packed.bin"],
        "combined wire SHA differs",
    )
    rows = [_object(row) for row in _list(producer["gallery_rows"])]
    ordinals = [_count(row["panel_ordinal"]) for row in rows]
    _require(
        [
            _count(ordinal)
            for batch in _list(producer["gallery_batches"])
            for ordinal in _list(_object(batch)["panel_ordinals"])
        ]
        == ordinals,
        "gallery batch membership differs",
    )
    ids = [_string(row["relative_path"]) for row in rows]
    _require(len(set(ids)) == len(ids), "unique gallery IDs required")
    gallery = b"".join(combined_wire[p * 130 : (p + 1) * 130] for p in ordinals)
    _require(len(gallery) == len(rows) * 130, "gallery wire length differs")
    ordered_ids = _canonical(ids)
    provenance = _canonical(
        {
            "schema": "connected-gallery-wire-selection-v1",
            "producer": producer,
            "gallery_wire_sha256": hashlib.sha256(gallery).hexdigest(),
            "ordered_ids_sha256": hashlib.sha256(ordered_ids).hexdigest(),
        }
    )
    return {
        "origin.json": bundle,
        "origin-export.json": receipt,
        "origin-owners.json": ownership_audit,
        "gallery.bin": gallery,
        "gallery-ids.json": ordered_ids,
        "gallery-provenance.json": provenance,
    }
