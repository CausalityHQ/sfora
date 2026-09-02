from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from sfora.data import ImageExample
from sfora.pass209_m4 import M4Example

_SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_pass209_m4.py"
_SPEC = importlib.util.spec_from_file_location("analyze_pass209_m4", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _argv(tmp_path: Path) -> list[str]:
    args: list[str] = []
    for prefix in ("dinov2", "siglip2", "selecting"):
        args.extend((f"--{prefix}-receipt", str(tmp_path / f"{prefix}-receipt.json")))
        args.extend((f"--{prefix}-descriptor", str(tmp_path / f"{prefix}.bin")))
        args.extend((f"--{prefix}-queries", str(tmp_path / f"{prefix}-queries.json")))
    args.extend(("--error-manifest", str(tmp_path / "errors.json")))
    for seed in (17, 29, 43):
        args.extend(("--m3-seed-receipt", str(tmp_path / f"seed-{seed}.json")))
    args.extend(
        (
            "--m3-aggregate",
            str(tmp_path / "aggregate.json"),
            "--m4-output",
            str(tmp_path / "m4.json"),
            "--adapter-output",
            str(tmp_path / "adapter.json"),
            "--source-revision",
            "1" * 40,
            "--execute",
        )
    )
    return args


def test_analyzer_cli_is_closed_and_requires_three_seed_receipts(tmp_path: Path) -> None:
    args = _MODULE.parse_args(_argv(tmp_path))
    assert args.execute is True
    assert len(args.m3_seed_receipt) == 3

    argv = _argv(tmp_path)
    index = argv.index("--m3-seed-receipt")
    del argv[index : index + 2]
    with pytest.raises(SystemExit):
        _MODULE.parse_args(argv)
    with pytest.raises(SystemExit):
        _MODULE.parse_args([*_argv(tmp_path), "--test-split", "test"])


def test_m3_loader_requires_aggregate_recomputed_from_exact_seed_bytes(
    tmp_path: Path,
) -> None:
    seed_paths = tuple(tmp_path / f"seed-{seed}.json" for seed in (17, 29, 43))
    for seed, path in zip((17, 29, 43), seed_paths, strict=True):
        path.write_bytes(f"seed-{seed}\n".encode())

    def aggregate(payloads: tuple[bytes, ...]) -> bytes:
        assert payloads == tuple(path.read_bytes() for path in seed_paths)
        return (
            json.dumps(
                {
                    "memorization_to_transfer_ratios": [0.2, 0.3, 0.35],
                    "schema": "fixture-m3",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()

    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_bytes(aggregate(tuple(path.read_bytes() for path in seed_paths)))
    value, state = _MODULE.load_m3_state(seed_paths, aggregate_path, aggregator=aggregate)
    assert value["schema"] == "fixture-m3"
    assert state == "T-low"

    aggregate_path.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="differs"):
        _MODULE.load_m3_state(seed_paths, aggregate_path, aggregator=aggregate)


def test_control_aggregator_registers_dynamic_module_for_dataclasses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = tmp_path / "control.py"
    control.write_text(
        """import sys
from dataclasses import dataclass

if __name__ not in sys.modules:
    raise RuntimeError("dynamic module is not registered")

@dataclass(frozen=True)
class Receipt:
    value: int

def control_aggregate_receipt_bytes(payloads):
    return b\"aggregate\\n\"
"""
    )
    monkeypatch.setattr(_MODULE, "_CONTROL_SCRIPT", control)

    aggregate = _MODULE._control_aggregator()

    assert aggregate((b"seed\n",)) == b"aggregate\n"


def test_adapter_receipt_binds_all_inputs_and_threshold_equality() -> None:
    seeds = tuple(
        _MODULE.canonical_json_bytes(
            {
                "schema": "sfora-siglip-proxy-control-seed-v1",
                "claim_eligible": False,
                "seed": seed,
            }
        )
        for seed in (17, 29, 43)
    )
    payload = _MODULE.adapter_receipt_bytes(
        source_revision="1" * 40,
        m4_payload=b"m4\n",
        m3_seed_payloads=seeds,
        m3_aggregate_payload=b"aggregate\n",
        m3_state="T-low",
        reachable_p10=0.25,
        dominant_rescuable=True,
    )
    value = json.loads(payload)
    assert value["decision"] == "F4-TRANSFER"
    assert value["m4_sha256"] == hashlib.sha256(b"m4\n").hexdigest()
    assert [row["seed"] for row in value["m3_seed_receipts"]] == [17, 29, 43]
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")

    with pytest.raises(ValueError, match="seed order"):
        _MODULE.adapter_receipt_bytes(
            source_revision="1" * 40,
            m4_payload=b"m4\n",
            m3_seed_payloads=(seeds[1], seeds[0], seeds[2]),
            m3_aggregate_payload=b"aggregate\n",
            m3_state="T-low",
            reachable_p10=0.25,
            dominant_rescuable=True,
        )


def test_analyzer_run_wires_all_three_cells_rgb_m3_and_atomic_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _argv(tmp_path)
    for path in tuple(
        tmp_path / name
        for name in (
            "dinov2-receipt.json",
            "dinov2.bin",
            "dinov2-queries.json",
            "siglip2-receipt.json",
            "siglip2.bin",
            "siglip2-queries.json",
            "selecting-receipt.json",
            "selecting.bin",
            "selecting-queries.json",
            "errors.json",
        )
    ):
        path.write_bytes(b"fixture\n")
    seeds = tuple(
        _MODULE.canonical_json_bytes(
            {
                "schema": "sfora-siglip-proxy-control-seed-v1",
                "claim_eligible": False,
                "seed": seed,
            }
        )
        for seed in (17, 29, 43)
    )
    for seed, payload in zip((17, 29, 43), seeds, strict=True):
        (tmp_path / f"seed-{seed}.json").write_bytes(payload)
    aggregate = _MODULE.canonical_json_bytes(
        {
            "memorization_to_transfer_ratios": [0.2, 0.3, 0.35],
            "schema": "fixture-m3",
        }
    )
    (tmp_path / "aggregate.json").write_bytes(aggregate)

    rows = [
        ImageExample(
            example_id=f"q{index}",
            image=Image.new("RGB", (2, 2), color=(index, 0, 0)),
            label=82 + index // 2,
        )
        for index in range(4)
    ]
    legacy_digest = hashlib.sha256(
        _MODULE.canonical_json_bytes(
            {"examples": sorted((str(row.example_id), int(row.label)) for row in rows)}
        )
    ).hexdigest()
    ordered_digest = hashlib.sha256(
        _MODULE.canonical_json_bytes(
            {"examples": [(str(row.example_id), int(row.label)) for row in rows]}
        )
    ).hexdigest()
    receipt_values = tuple(
        SimpleNamespace(
            spec=SimpleNamespace(cell=cell),
            receipt={
                "source_revision": "1" * 40,
                "dataset_revision": "2" * 40,
                "dataset_examples_sha256": legacy_digest,
                "dataset_examples_ordered_sha256": ordered_digest,
            },
        )
        for cell in ("dinov2-large", "siglip2-so400m", "siglip-so400m")
    )
    receipts: tuple[Any, Any, Any] = (
        receipt_values[0],
        receipt_values[1],
        receipt_values[2],
    )
    observed: dict[str, Any] = {}

    def load_cells(
        paths: tuple[Any, ...], *, expected_examples: tuple[M4Example, ...]
    ) -> tuple[Any, Any, Any]:
        observed["cell_paths"] = tuple(path.receipt.name for path in paths)
        observed["examples"] = expected_examples
        return receipts

    manifest = {
        "dataset_revision": "2" * 40,
        "dataset_examples_sha256": legacy_digest,
    }
    source_errors = tuple(SimpleNamespace(query_position=index) for index in range(103))

    def serialize_m4(**kwargs: Any) -> bytes:
        observed["m4_cells"] = kwargs["cells"]
        observed["rgb_count"] = len(kwargs["rgb_sha256"])
        return bytes(
            _MODULE.canonical_json_bytes(
                {
                    "objective": {
                        "bootstrap": {"p10": 0.25},
                        "dominant_pair_rescuable": False,
                    },
                    "schema": "sfora-pass209-m4-objective-rescue-v1",
                }
            )
        )

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(_MODULE, "SUBSTRATE_F0_CLASSES", {82, 83})
    monkeypatch.setattr(_MODULE, "load_image_retrieval_examples", lambda **_: rows)
    monkeypatch.setattr(_MODULE, "materialize_image", lambda image: image)
    monkeypatch.setattr(_MODULE, "load_m4_cells", load_cells)
    monkeypatch.setattr(
        _MODULE,
        "load_m4_source_errors",
        lambda *_args, **_kwargs: (manifest, source_errors),
    )
    monkeypatch.setattr(_MODULE, "m4_receipt_bytes", serialize_m4)
    monkeypatch.setattr(_MODULE, "_control_aggregator", lambda: lambda _: aggregate)

    m4_payload, adapter_payload = _MODULE._run(_MODULE.parse_args(argv))
    assert observed["cell_paths"] == (
        "dinov2-receipt.json",
        "siglip2-receipt.json",
        "selecting-receipt.json",
    )
    assert observed["m4_cells"] == receipts
    assert observed["rgb_count"] == 4
    assert len(observed["examples"]) == 4
    assert (tmp_path / "m4.json").read_bytes() == m4_payload
    assert (tmp_path / "adapter.json").read_bytes() == adapter_payload
    assert json.loads(adapter_payload)["decision"] == "F4-TRANSFER"
