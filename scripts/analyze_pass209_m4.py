#!/usr/bin/env python3
"""Authenticate Pass209 M4 evidence and emit the frozen family decision."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from PIL import Image

from sfora.data import load_image_retrieval_examples, materialize_image
from sfora.pass209_m4 import (
    M4CellPaths,
    M4Example,
    adapt_m3_m4,
    canonical_json_bytes,
    classify_m3_transfer,
    load_m4_cells,
    load_m4_source_errors,
    m4_receipt_bytes,
    publish_new_outputs,
    rgb_record_sha256,
)
from sfora.substrate_screen import SUBSTRATE_F0_CLASSES

_ERROR_MANIFEST_SHA256 = "64d491607d4dac144b31edac3a182130e6f94f994a272f612c195a7a72d55611"
_CONTROL_SCRIPT = Path(__file__).with_name("run_siglip_proxy_control.py")


def _revision(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise argparse.ArgumentTypeError("expected an exact revision")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for prefix in ("dinov2", "siglip2", "selecting"):
        parser.add_argument(f"--{prefix}-receipt", type=Path, required=True)
        parser.add_argument(f"--{prefix}-descriptor", type=Path, required=True)
        parser.add_argument(f"--{prefix}-queries", type=Path, required=True)
    parser.add_argument("--error-manifest", type=Path, required=True)
    parser.add_argument("--m3-seed-receipt", type=Path, action="append", required=True)
    parser.add_argument("--m3-aggregate", type=Path, required=True)
    parser.add_argument("--m4-output", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--source-revision", type=_revision, required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args(argv)
    if len(args.m3_seed_receipt) != 3:
        parser.error("exactly three --m3-seed-receipt paths are required")
    return args


def _require_offline() -> None:
    required = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
    if any(os.environ.get(name) != "1" for name in required):
        raise RuntimeError("Pass209 M4 analyzer requires the offline environment")


def _control_aggregator() -> Callable[[tuple[bytes, ...]], bytes]:
    spec = importlib.util.spec_from_file_location("pass209_control_runner", _CONTROL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("M3 control aggregator is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "control_aggregate_receipt_bytes", None)
    if not callable(function):
        raise RuntimeError("M3 control aggregator is unavailable")
    return cast(Callable[[tuple[bytes, ...]], bytes], function)


def load_m3_state(
    seed_paths: tuple[Path, Path, Path],
    aggregate_path: Path,
    *,
    aggregator: Callable[[tuple[bytes, ...]], bytes] | None = None,
) -> tuple[dict[str, object], str]:
    seed_payloads = cast(
        tuple[bytes, bytes, bytes],
        tuple(path.read_bytes() for path in seed_paths),
    )
    actual = aggregate_path.read_bytes()
    return _authenticate_m3_state(
        seed_payloads,
        actual,
        aggregator=aggregator,
    )


def _authenticate_m3_state(
    seed_payloads: tuple[bytes, bytes, bytes],
    aggregate_payload: bytes,
    *,
    aggregator: Callable[[tuple[bytes, ...]], bytes] | None = None,
) -> tuple[dict[str, object], str]:
    expected = (aggregator or _control_aggregator())(seed_payloads)
    actual = aggregate_payload
    if actual != expected:
        raise ValueError("M3 aggregate differs from authenticated seed receipts")
    try:
        value = json.loads(actual)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("M3 aggregate is not JSON") from error
    if type(value) is not dict:
        raise ValueError("M3 aggregate schema differs")
    ratios = value.get("memorization_to_transfer_ratios")
    if type(ratios) is not list or len(ratios) != 3:
        raise ValueError("M3 aggregate schema differs")
    return value, classify_m3_transfer(tuple(ratios))


def adapter_receipt_bytes(
    *,
    source_revision: str,
    m4_payload: bytes,
    m3_seed_payloads: tuple[bytes, bytes, bytes],
    m3_aggregate_payload: bytes,
    m3_state: str,
    reachable_p10: float,
    dominant_rescuable: bool,
) -> bytes:
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError("adapter source revision differs")
    seed_values: list[int] = []
    for payload in m3_seed_payloads:
        try:
            seed_receipt = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("adapter seed receipt is not JSON") from error
        seed = seed_receipt.get("seed") if type(seed_receipt) is dict else None
        if (
            type(seed_receipt) is not dict
            or canonical_json_bytes(seed_receipt) != payload
            or seed_receipt.get("schema") != "sfora-siglip-proxy-control-seed-v1"
            or seed_receipt.get("claim_eligible") is not False
            or type(seed) is not int
        ):
            raise ValueError("adapter seed receipt authority differs")
        seed_values.append(seed)
    if tuple(seed_values) != (17, 29, 43):
        raise ValueError("adapter seed order differs")
    decision = adapt_m3_m4(
        m3_state=m3_state,
        reachable_p10=reachable_p10,
        dominant_rescuable=dominant_rescuable,
    )
    value: dict[str, object] = {
        "schema": "sfora-pass209-m3-m4-family-adapter-v1",
        "claim_eligible": False,
        "source_revision": source_revision,
        "m4_sha256": hashlib.sha256(m4_payload).hexdigest(),
        "m3_aggregate_sha256": hashlib.sha256(m3_aggregate_payload).hexdigest(),
        "m3_seed_receipts": [
            {"seed": seed, "sha256": hashlib.sha256(payload).hexdigest()}
            for seed, payload in zip(seed_values, m3_seed_payloads, strict=True)
        ],
        "m3_state": m3_state,
        "reachable_p10": reachable_p10,
        "dominant_pair_rescuable": dominant_rescuable,
        "decision": decision,
    }
    return canonical_json_bytes(value)


def _run(args: argparse.Namespace) -> tuple[bytes, bytes]:
    _require_offline()
    outputs = (args.m4_output, args.adapter_output)
    if args.m4_output.resolve() == args.adapter_output.resolve():
        raise ValueError("M4 output paths must be distinct")
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite an M4 output")
    cell_paths = tuple(
        M4CellPaths(
            receipt=getattr(args, f"{prefix}_receipt"),
            descriptor=getattr(args, f"{prefix}_descriptor"),
            queries=getattr(args, f"{prefix}_queries"),
        )
        for prefix in ("dinov2", "siglip2", "selecting")
    )
    manifest, source_errors = load_m4_source_errors(
        args.error_manifest,
        expected_sha256=_ERROR_MANIFEST_SHA256,
    )
    all_examples = load_image_retrieval_examples(dataset_name="cars", split="train")
    legacy_rows = sorted((str(row.example_id), int(row.label)) for row in all_examples)
    legacy_digest = hashlib.sha256(canonical_json_bytes({"examples": legacy_rows})).hexdigest()
    holdout = tuple(row for row in all_examples if int(row.label) in SUBSTRATE_F0_CLASSES)
    examples = tuple(
        M4Example(position=index, example_id=str(row.example_id), label=int(row.label))
        for index, row in enumerate(holdout)
    )
    ordered_digest = hashlib.sha256(
        canonical_json_bytes({"examples": [(row.example_id, row.label) for row in examples]})
    ).hexdigest()
    cells = load_m4_cells(cell_paths, expected_examples=examples)
    if cells[0].receipt.get("source_revision") != args.source_revision:
        raise ValueError("M4 analyzer source revision differs")
    for cell in cells:
        for key in ("dataset_revision", "dataset_examples_sha256"):
            if cell.receipt.get(key) != manifest.get(key):
                raise ValueError(f"M4 manifest {key} binding differs")
    if legacy_digest != cells[0].receipt["dataset_examples_sha256"]:
        raise ValueError("M4 dataset example-set authority differs")
    if ordered_digest != cells[0].receipt["dataset_examples_ordered_sha256"]:
        raise ValueError("M4 ordered example authority differs")
    rgb_digests = []
    for row in holdout:
        image = materialize_image(row.image)
        if not isinstance(image, Image.Image):
            raise TypeError("M4 image materialization did not produce PIL")
        rgb_digests.append(rgb_record_sha256(image.convert("RGB")))
    m4_payload = m4_receipt_bytes(
        cells=cells,
        source_errors=source_errors,
        error_manifest_sha256=_ERROR_MANIFEST_SHA256,
        examples=examples,
        rgb_sha256=tuple(rgb_digests),
    )
    m4_value = json.loads(m4_payload)
    bootstrap = m4_value["objective"]["bootstrap"]
    dominant_rescuable = m4_value["objective"]["dominant_pair_rescuable"]
    if type(bootstrap) is not dict:
        raise ValueError("M4 bootstrap result differs")
    seed_paths = tuple(args.m3_seed_receipt)
    assert len(seed_paths) == 3
    seed_payloads = cast(
        tuple[bytes, bytes, bytes],
        tuple(path.read_bytes() for path in seed_paths),
    )
    m3_aggregate_payload = args.m3_aggregate.read_bytes()
    m3_value, m3_state = _authenticate_m3_state(
        seed_payloads,
        m3_aggregate_payload,
    )
    del m3_value
    adapter_payload = adapter_receipt_bytes(
        source_revision=args.source_revision,
        m4_payload=m4_payload,
        m3_seed_payloads=seed_payloads,
        m3_aggregate_payload=m3_aggregate_payload,
        m3_state=m3_state,
        reachable_p10=bootstrap["p10"],
        dominant_rescuable=dominant_rescuable,
    )
    publish_new_outputs(((args.m4_output, m4_payload), (args.adapter_output, adapter_payload)))
    return m4_payload, adapter_payload


def main() -> None:
    args = parse_args()
    m4_payload, adapter_payload = _run(args)
    print(
        json.dumps(
            {
                "m4_sha256": hashlib.sha256(m4_payload).hexdigest(),
                "adapter_sha256": hashlib.sha256(adapter_payload).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
