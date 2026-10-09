"""Bind serving declarations to independently pinned, opaque fragment bytes.

Payload sizes remain expected declarations; this does not authenticate payload
files, installed sources, native execution, serving quality or speed.
"""

import hashlib

from .connected_gallery_provenance import (
    JSONObject,
    _canonical,
    _count,
    _list,
    _object,
    _parse,
    _require,
    _same,
    _sha,
    _string,
    bind_gallery_provenance,
)

_FRAGMENT = (
    "origin.json",
    "origin-export.json",
    "origin-owners.json",
    "gallery.bin",
    "gallery-ids.json",
    "gallery-provenance.json",
)
_PAYLOAD = ("vision.pt", "endpoint.pt", "processor.json")


def _json_bytes(raw: bytes) -> None:
    _require(len(raw) <= 64 * 1024 * 1024, "JSON exceeds 64 MiB")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("UTF-8 JSON required") from error
    # json.loads(bytes) also accepts UTF-16/32, whose syntax contains raw NULs.
    _require("\0" not in text, "UTF-8 JSON required")


def bind_serving_manifest(
    serving: bytes,
    fragment: dict[str, bytes],
    *,
    trusted_serving_sha256: str,
    trusted_fragment_sha256: dict[str, str],
) -> JSONObject:
    """Authenticate every input before JSON, returning fresh manifest/producer data.

    Pins must come from independent caller authority. Owner bytes stay opaque.
    Payload SHAs bind to the original bundle; their sizes require later actual-file
    admission. encoder_binding_sha256 is a new canonical association digest, not
    an original typed tensor fingerprint. Invalid inputs raise ValueError.
    """
    _require(type(serving) is bytes, "immutable serving bytes required")
    _require(
        type(fragment) is dict and type(trusted_fragment_sha256) is dict,
        "builtin fragment and pin dictionaries required",
    )
    # Keep authenticated immutable bytes/pins stable across caller mapping changes.
    fragment = fragment.copy()
    trusted_fragment_sha256 = trusted_fragment_sha256.copy()
    for mapping in (fragment, trusted_fragment_sha256):
        for name in mapping:
            _string(name)
    names = set(_FRAGMENT)
    _require(
        fragment.keys() == trusted_fragment_sha256.keys() == names,
        "exact fragment and pin names required",
    )
    _require(
        all(type(raw) is bytes for raw in fragment.values()), "immutable fragment bytes required"
    )
    _require(
        hashlib.sha256(serving).hexdigest() == _sha(trusted_serving_sha256),
        "trusted serving SHA differs",
    )
    observed = {}
    for name in _FRAGMENT:
        observed[name] = hashlib.sha256(fragment[name]).hexdigest()
        _require(
            observed[name] == _sha(trusted_fragment_sha256[name]),
            f"trusted {name} SHA differs",
        )
    for raw in (
        serving,
        fragment["origin.json"],
        fragment["origin-export.json"],
        fragment["gallery-ids.json"],
        fragment["gallery-provenance.json"],
    ):
        _json_bytes(raw)
    manifest = _object(_parse(serving), {"schema", "files", "origin", "gallery", "request"})
    _require(manifest["schema"] == "siglip2-connected-mlp-serving-v2", "serving schema differs")
    _same(
        manifest["request"],
        {
            "device": "cuda",
            "batch_min": 1,
            "batch_max": 32,
            "dimensions": 128,
            "k": 10,
            "ties": "ordinal-ascending",
        },
        "request contract differs",
    )
    producer = bind_gallery_provenance(
        fragment["origin-export.json"],
        fragment["origin.json"],
        trusted_receipt_sha256=trusted_fragment_sha256["origin-export.json"],
        trusted_bundle_sha256=trusted_fragment_sha256["origin.json"],
    )
    identity = _object(producer["identity"])
    _same(
        manifest["origin"],
        {
            "bundle_sha256": trusted_fragment_sha256["origin.json"],
            "endpoint_state_sha256": identity["inference_state_sha256"],
            "fixed_sha256": identity["fixed_sha256"],
        },
        "original identity differs",
    )
    files = _object(manifest["files"], names | set(_PAYLOAD))
    payload_files = _object(identity["bundle_files"])
    for name, value in files.items():
        fact = _object(value, {"path", "bytes", "sha256"})
        _require(_string(fact["path"]) == name, "prescribed basename required")
        size, digest = _count(fact["bytes"]), _sha(fact["sha256"])
        if name in names:
            _require(size == len(fragment[name]), "fragment size differs")
            _require(digest == observed[name], "fragment SHA differs")
        else:
            _require(digest == payload_files[name], "original payload SHA differs")
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
    _require(
        fragment["gallery-ids.json"] == _canonical(ids), "canonical ordered gallery IDs differ"
    )
    _require(len(rows) >= 10, "at least ten gallery rows required")
    _require(len(fragment["gallery.bin"]) == len(rows) * 130, "gallery wire length differs")
    _require(
        fragment["gallery-provenance.json"]
        == _canonical(
            {
                "schema": "connected-gallery-wire-selection-v1",
                "producer": producer,
                "gallery_wire_sha256": observed["gallery.bin"],
                "ordered_ids_sha256": observed["gallery-ids.json"],
            }
        ),
        "canonical gallery selection differs",
    )
    _same(
        manifest["gallery"],
        {
            "count": len(rows),
            "dimensions": 128,
            "bytes_per_row": 130,
            "wire_sha256": observed["gallery.bin"],
            "ordered_ids_sha256": observed["gallery-ids.json"],
            "provenance_sha256": observed["gallery-provenance.json"],
            "encoder_binding_sha256": hashlib.sha256(
                _canonical(
                    {key: producer[key] for key in ("identity", "gallery_batches", "gallery_rows")}
                )
            ).hexdigest(),
        },
        "gallery association differs",
    )
    return {"manifest": manifest, "producer": producer}
