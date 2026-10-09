"""Bind accepted original producer metadata without reading any original path.

The caller must independently trust both original byte hashes. Receipt flags
are observations, never acceptance authority. This does not verify payloads,
image/output equivalence, native qualification, serving quality or speed.
Only combined wire hashes exist: gallery_wire_sha256 is always None.
"""

import hashlib
import json
import math
import re
from typing import cast

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
type JSONObject = dict[str, JSONValue]

_SCOPE = {
    "arm": "control",
    "manifest_sha256": "55cde4ef9de3c3636da2215c706115f36b90436f47fc23b345f72cc168874726",
    "arm_sha256": "1f3ad34bbd20a3b375ccb4908f9a3da05b63b514395cb553e1c81925789b2280",
}
_MEMBERS = set(
    [
        "config",
        "buffers",
        "processor",
        "head",
        "A",
        "means",
        "C",
        "mu_train",
        "mu_train_provenance",
        "scope",
        "common_statistics",
        "base_vision",
        "encoder",
        "encoder_identity",
        "arm",
    ]
)
_ROW_KEYS = {
    "image_sha256",
    "original_row",
    "panel_ordinal",
    "path",
    "product",
    "relative_path",
    "role",
    "target",
    "train_row",
}
_WITNESSES = {"rgb_sha256", "pixels_sha256", "outputs_sha256"}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _object(value: object, keys: set[str] | None = None) -> JSONObject:
    _require(type(value) is dict, "builtin JSON object required")
    result = cast(JSONObject, value)
    if keys is not None:
        _require(result.keys() == keys, "exact object keys differ")
    return result


def _list(value: object) -> list[JSONValue]:
    _require(type(value) is list, "builtin JSON list required")
    return cast(list[JSONValue], value)


def _string(value: object) -> str:
    _require(type(value) is str and bool(value), "nonempty builtin string required")
    return cast(str, value)


def _sha(value: object) -> str:
    result = _string(value)
    _require(re.fullmatch("[0-9a-f]{64}", result) is not None, "exact SHA256 required")
    return result


def _count(value: object) -> int:
    _require(type(value) is int and value >= 0, "nonnegative builtin count required")
    return cast(int, value)


def _path(value: object, *, absolute: bool) -> str:
    result = _string(value)
    _require(
        result.startswith("/") is absolute and "\\" not in result and "\0" not in result,
        "canonical POSIX path required",
    )
    parts = result[1:].split("/") if absolute else result.split("/")
    _require(all(p not in {"", ".", ".."} for p in parts), "canonical path components required")
    return result


def _file(value: object) -> JSONObject:
    result = _object(value, {"path", "sha256"})
    _path(result["path"], absolute=True)
    _sha(result["sha256"])
    return result


def _canonical(value: object) -> bytes:
    # Exact reference.json_digest definition; not the trainer's typed fingerprint.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _same(left: object, right: object, message: str) -> None:
    _require(_canonical(left) == _canonical(right), message)


def _pairs(pairs: list[tuple[str, JSONValue]]) -> JSONObject:
    result: JSONObject = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError("nonfinite JSON constant: " + value)


def _float(value: str) -> float:
    result = float(value)
    _require(math.isfinite(result), "nonfinite JSON number")
    return result


def _parse(raw: bytes) -> JSONObject:
    try:
        return _object(
            json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("strict JSON required") from error


def _identity(record: JSONObject, bundle: JSONObject, bundle_sha: str) -> JSONObject:
    _object(
        bundle,
        {
            "schema",
            "endpoint_state_sha256",
            "base_vision_sha256",
            "vision_sha256",
            "encoder_identity",
            "scope",
            "files",
            "code",
            "environment",
        },
    )
    _require(bundle["schema"] == "siglip2-connected-mlp-bundle-v1", "bundle schema differs")
    facts = _object(
        record.get("payload_facts"),
        {
            "identity",
            "members",
            "vision_sha256",
            "base_vision_sha256",
            "fixed_sha256",
            "processor_config_sha256",
            "terminal_state_sha256",
            "inference_state_sha256",
            "bundle",
        },
    )
    identity = _object(facts["identity"])
    members = _object(facts["members"], _MEMBERS)
    for value in members.values():
        _sha(value)
    for key in (
        "vision_sha256",
        "base_vision_sha256",
        "fixed_sha256",
        "processor_config_sha256",
        "terminal_state_sha256",
        "inference_state_sha256",
    ):
        _sha(facts[key])
    binding = _object(record.get("binding"))
    _require(
        binding.get("stage") == "full" and binding.get("panel") == "selection",
        "bound export stage/panel differs",
    )
    endpoints = [_object(e) for e in _list(binding.get("endpoints"))]
    matches = [e for e in endpoints if e.get("arm") == "control" and e.get("seed") == 179061]
    _require(len(matches) == 1, "exactly one bound control-179061 endpoint required")
    endpoint = matches[0]
    _same(
        binding["endpoints"],
        _object(record.get("launch")).get("endpoints"),
        "launch/bound endpoint differs",
    )
    _require(
        _count(endpoint["seed"]) == _count(identity.get("seed")) == 179061
        and identity.get("arm") == "control",
        "endpoint identity differs",
    )
    descriptor = _file(facts["bundle"])
    _same(descriptor, _file(endpoint.get("bundle")), "bound bundle differs")
    _require(descriptor["sha256"] == bundle_sha, "payload bundle SHA differs")
    _require(
        _sha(bundle["endpoint_state_sha256"])
        == facts["inference_state_sha256"]
        == _sha(record.get("inference_state_sha256"))
        == _sha(endpoint.get("inference_state_sha256")),
        "endpoint state differs",
    )
    _require(
        facts["terminal_state_sha256"] == _sha(endpoint.get("terminal_state_sha256")),
        "terminal state differs",
    )
    base = _object(identity.get("base_vision"), {"checkpoint", "sha256"})
    _require(
        _sha(bundle["vision_sha256"])
        == facts["vision_sha256"]
        == _sha(bundle["base_vision_sha256"])
        == facts["base_vision_sha256"]
        == _sha(base["sha256"]),
        "control vision/base identity differs",
    )
    encoder = _object(bundle["encoder_identity"])
    _same(encoder, _object(identity.get("encoder_identity")), "encoder identity differs")
    _require(_sha(encoder.get("buffers_sha256")) == members["buffers"], "encoder buffers differ")
    _same(bundle["scope"], _SCOPE, "original bundle scope differs")
    _same(identity.get("scope"), bundle["scope"], "identity scope differs")
    _require(
        _file(binding.get("scope"))["sha256"] == _SCOPE["manifest_sha256"], "bound scope differs"
    )
    _same(_object(record.get("source")), _object(identity.get("source")), "source differs")
    flags = _object(record.get("numerical_flags"))
    _same(flags, _object(identity.get("numerical_flags")), "numerical flags differ")
    for key in ("threads", "interop_threads"):
        if key in flags:
            _require(_count(flags[key]) > 0, "positive thread count required")
    files = _object(bundle["files"], {"endpoint.pt", "processor.json", "vision.pt"})
    _require(
        _sha(files["vision.pt"]) == _file(base["checkpoint"])["sha256"],
        "original vision file differs",
    )
    guards = _object(record.get("input_guards"))
    terminal = _object(endpoint.get("terminal"))
    for value in (
        endpoint.get("checkpoint"),
        endpoint.get("launch"),
        terminal.get("receipt"),
        terminal.get("log"),
        binding.get("scope"),
        base["checkpoint"],
    ):
        original = _file(value)
        _require(
            guards.get(_string(original["path"])) == original["sha256"],
            "original endpoint/base/scope file guard differs",
        )
    evaluator = _object(
        record.get("source_code"),
        {
            "evaluate_siglip2_connected_mlp.py",
            "test_connected_mlp_evaluation.py",
        },
    )
    argv = _list(_object(record.get("invocation")).get("argv"))
    _require(bool(argv), "original evaluator invocation required")
    evaluator_path = _path(argv[0], absolute=True)
    _require(
        evaluator_path.rsplit("/", 1)[-1] == "evaluate_siglip2_connected_mlp.py",
        "original evaluator path differs",
    )
    for name, digest in evaluator.items():
        _require(
            guards.get(evaluator_path.rsplit("/", 1)[0] + "/" + name) == _sha(digest),
            "original evaluator source guard differs",
        )
    bundle_path = _path(descriptor["path"], absolute=True)
    _require(
        bundle_path.rsplit("/", 1)[-1] == "bundle.json" and guards.get(bundle_path) == bundle_sha,
        "original bundle guard differs",
    )
    code = _object(bundle["code"])
    _require(not files.keys() & code.keys(), "bundle code/file names overlap")
    training = _object(
        _object(binding.get("training")).get("code"),
        {
            "train_siglip2_connected_mlp.py",
            "test_siglip2_connected_mlp.py",
        },
    )
    for name, digest in training.items():
        _require(code.get(name) == _sha(digest), "training source differs")
    for name, digest in {**files, **code}.items():
        _require("/" not in _path(name, absolute=False), "original bundle member name differs")
        path = bundle_path.rsplit("/", 1)[0] + "/" + name
        _require(guards.get(path) == _sha(digest), "original bundle file/source guard differs")
    return {**facts, "numerical_flags": flags, "bundle_files": files}


def _gallery(record: JSONObject) -> tuple[JSONObject, list[JSONValue], list[JSONValue]]:
    panel = _object(record.get("panel_facts"), {"raw", "unit", "codes", "inverse_norms"})
    raw_shape = _list(_object(panel["raw"]).get("shape"))
    _require(len(raw_shape) == 2, "panel shape differs")
    count = _count(raw_shape[0])
    _require(count > 0, "nonempty panel required")
    for name, dtype in (
        ("raw", "torch.float32"),
        ("unit", "torch.float32"),
        ("codes", "torch.int8"),
        ("inverse_norms", "torch.float16"),
    ):
        fact = _object(panel[name], {"shape", "dtype", "sha256"})
        shape = [_count(n) for n in _list(fact["shape"])]
        _require(
            shape == ([count] if name == "inverse_norms" else [count, 128])
            and fact["dtype"] == dtype,
            "panel shape/dtype differs",
        )
        _sha(fact["sha256"])
    wires = _object(
        record.get("files"),
        {
            "control-179061.raw.npy",
            "control-179061.unit.npy",
            "control-179061.packed.bin",
        },
    )
    for digest in wires.values():
        _sha(digest)
    sizes = _object(record.get("batch_sizes"), {"query", "gallery"})
    images = [
        _object(b, _WITNESSES | {"rows", "role", "residual_oracle"})
        for b in _list(record.get("images"))
    ]
    all_rows: dict[int, JSONObject] = {}
    gallery_rows: list[JSONValue] = []
    gallery_batches: list[JSONValue] = []
    position = 0
    for role in ("query", "gallery"):
        recorded = [_count(n) for n in _list(sizes[role])]
        _require(
            bool(recorded) and all(n == 32 for n in recorded[:-1]) and 1 <= recorded[-1] <= 32,
            "complete B32 role/tail sizes required",
        )
        previous = -1
        for index, size in enumerate(recorded):
            _require(position < len(images), "missing image batch")
            batch = images[position]
            position += 1
            _require(batch["role"] == role, "query then gallery batches required")
            rows = [_object(r, _ROW_KEYS) for r in _list(batch["rows"])]
            _require(len(rows) == size, "recorded batch size differs")
            ordinals: list[JSONValue] = []
            for row in rows:
                ordinal = _count(row["panel_ordinal"])
                _require(
                    previous < ordinal < count and ordinal not in all_rows,
                    "unique ascending complete panel ordinals required",
                )
                _require(row["role"] == role, "row role differs")
                for key in ("original_row", "target", "train_row"):
                    _count(row[key])
                _sha(row["image_sha256"])
                _path(row["path"], absolute=True)
                _path(row["relative_path"], absolute=False)
                _string(row["product"])
                previous = ordinal
                all_rows[ordinal] = row
                ordinals.append(ordinal)
            for key in _WITNESSES:
                _sha(batch[key])
            oracle = batch["residual_oracle"]
            if index in (0, len(recorded) - 1):
                _same(
                    oracle,
                    {
                        "C_exact_zero": False,
                        "residual_nonzero_witness": True,
                        "omitted_C_mutant_rejected": True,
                        "wrong_mu_mutant_rejected": True,
                    },
                    "first/last residual oracle differs",
                )
            else:
                _require(oracle is None, "unexpected interior residual oracle")
            if role == "gallery":
                gallery_rows.extend(rows)
                gallery_batches.append(
                    {
                        **{k: v for k, v in batch.items() if k != "rows"},
                        "panel_ordinals": ordinals,
                    }
                )
    _require(
        position == len(images) and len(all_rows) == count, "complete panel membership differs"
    )
    ordered = [all_rows[i] for i in range(count)]
    _require(
        hashlib.sha256(_canonical(ordered)).hexdigest()
        == _sha(record.get("ordered_images_sha256")),
        "ordered images SHA differs",
    )
    combined: JSONObject = {
        "files": wires,
        "count": count,
        "dimensions": 128,
        "bytes_per_row": 130,
        "panel_facts": panel,
    }
    return combined, gallery_batches, gallery_rows


def bind_gallery_provenance(
    receipt: bytes,
    bundle: bytes,
    *,
    trusted_receipt_sha256: str,
    trusted_bundle_sha256: str,
) -> JSONObject:
    """Return an inert association of two independently accepted original inputs.

    Both pins are checked before either parse. Full payload facts and original
    bundle file hashes stay distinct under identity. Combined wire metadata
    covers BOTH roles; gallery batches preserve ordinal membership and original
    witnesses, and gallery_rows preserves every original row key and value.
    No new identity/ID/partition digest or gallery wire hash is constructed.
    Invalid bytes, schemas or correspondences raise ValueError.
    """
    _require(type(receipt) is bytes and type(bundle) is bytes, "immutable bytes required")
    _require(
        hashlib.sha256(receipt).hexdigest() == _sha(trusted_receipt_sha256),
        "trusted receipt SHA differs",
    )
    _require(
        hashlib.sha256(bundle).hexdigest() == _sha(trusted_bundle_sha256),
        "trusted bundle SHA differs",
    )
    record, manifest = _parse(receipt), _parse(bundle)
    _require(
        record.get("schema") == "siglip2-connected-mlp-evaluation-v1"
        and record.get("phase") == "export"
        and record.get("stage") == "full"
        and record.get("panel") == "selection"
        and record.get("arm") == "control"
        and _count(record.get("seed")) == 179061,
        "original export role/schema differs",
    )
    identity = _identity(record, manifest, trusted_bundle_sha256)
    combined, batches, rows = _gallery(record)
    return {
        "schema": "connected-gallery-producer-binding-v1",
        "original_receipt_sha256": trusted_receipt_sha256,
        "original_bundle_sha256": trusted_bundle_sha256,
        "identity": identity,
        "combined_wire": combined,
        "ordered_images_sha256": record["ordered_images_sha256"],
        "gallery_batches": batches,
        "gallery_rows": rows,
        "gallery_wire_sha256": None,
    }
