from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from PIL import Image

from scripts import observe_prism_cue_panel as observer_cli
from scripts.observe_prism_cue_panel import observe_prism_cue_panel, parse_args
from sfora.pass209_m4 import canonical_json_bytes
from sfora.prism_measurement import PRISM_CHANNELS, PrismObservationCapabilityRow
from sfora.prism_observer import (
    PrismChannelPrompt,
    PrismPayloadAuthority,
    PrismPromptBundle,
    derive_prism_token_protocol,
    validate_prism_completion_bundle_bytes,
)


def _prompt_bundle() -> PrismPromptBundle:
    rows = []
    for channel in PRISM_CHANNELS:
        prompt = (
            f"channel={channel}; compare the two anonymous vehicle images using only "
            "the named visual cue. Return the registered compact completion grammar."
        )
        rows.append(
            PrismChannelPrompt(
                channel=channel,
                prompt_utf8=prompt,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                max_new_tokens=192,
                temperature_ppm=1_000_000,
                top_p_ppm=1_000_000,
            )
        )
    return PrismPromptBundle("sfora-prism-prompt-bundle-v1", tuple(rows))


class _Tokenizer:
    def __init__(self) -> None:
        literals = (
            *(f"channel={channel};" for channel in PRISM_CHANNELS),
            "left_visible=yes;right_visible=yes;",
            "left_visible=yes;right_visible=no;",
            "left_visible=no;right_visible=yes;",
            "left_visible=no;right_visible=no;",
            "relation=same;",
            "relation=different;",
            "relation=indeterminate;",
            "confidence=low;evidence_left=",
            "confidence=medium;evidence_left=",
            "confidence=high;evidence_left=",
            ";evidence_right=",
            "<PRISM_END>",
        )
        self.mapping = {text: (index + 1,) for index, text in enumerate(literals)}
        self.all_special_ids: tuple[int, ...] = ()

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(self.mapping[text])

    def decode(self, ids: list[int], **_: object) -> str:
        target = tuple(ids)
        return next(text for text, value in self.mapping.items() if value == target)


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()


class _Adapter:
    def __init__(self) -> None:
        self.prepared: list[tuple[tuple[tuple[int, int, int], ...], str, tuple[int, int], int]] = []
        self.generated: list[tuple[int, float, float, int]] = []

    def prepare_image_pair(
        self,
        images: tuple[object, object],
        prompt_utf8: str,
        attribute_token_span: tuple[int, int],
        patch_tokens_per_image: int,
    ) -> int:
        colors = tuple(tuple(image[0, 0]) for image in images)  # type: ignore[index]
        self.prepared.append(
            (colors, prompt_utf8, attribute_token_span, patch_tokens_per_image)
        )
        return len(self.prepared)

    def generate(
        self,
        pair: object,
        seed: int,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> tuple[int, ...]:
        self.generated.append((seed, temperature, top_p, max_new_tokens))
        return (int(pair), seed & 0xFFFF)


def _payloads(root: Path) -> tuple[tuple[PrismPayloadAuthority, ...], tuple[str, ...]]:
    root.mkdir()
    authorities = []
    digests = []
    for ordinal in range(64):
        image = Image.new("RGB", (2, 2), (ordinal, 7, 11))
        stream = io.BytesIO()
        image.save(stream, format="PNG", compress_level=9, optimize=False)
        raw = stream.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        (root / f"{digest}.png").write_bytes(raw)
        authorities.append(PrismPayloadAuthority(digest, len(raw), 2, 2, "RGB"))
        digests.append(digest)
    return tuple(authorities), tuple(digests)


def _capability(digests: tuple[str, ...]) -> tuple[PrismObservationCapabilityRow, ...]:
    rows = []
    for pair in range(32):
        for channel in PRISM_CHANNELS:
            rows.append(
                PrismObservationCapabilityRow(
                    pair_handle=f"{pair + 1:064x}",
                    channel=channel,
                    left_payload_sha256=digests[pair * 2],
                    right_payload_sha256=digests[pair * 2 + 1],
                    left_first=pair % 2 == 0,
                    generation_seed=10_000 + len(rows),
                )
            )
    return tuple(rows)


def test_observer_loads_once_and_generates_each_anonymous_row_once(tmp_path: Path) -> None:
    payload_dir = tmp_path / "payloads"
    authorities, digests = _payloads(payload_dir)
    capability = _capability(digests)
    prompts = _prompt_bundle()
    processor = _Processor()
    protocol = derive_prism_token_protocol(processor, prompts)
    protocol_sha256 = hashlib.sha256(canonical_json_bytes(asdict(protocol))).hexdigest()
    adapter = _Adapter()
    loads = 0

    def load() -> tuple[_Adapter, _Processor]:
        nonlocal loads
        loads += 1
        return adapter, processor

    output = tmp_path / "completion.json"
    progress = tmp_path / "progress.json"
    observe_prism_cue_panel(
        output,
        progress,
        phase="diagnostic",
        capability=capability,
        prompt_bundle=prompts,
        payload_authorities=authorities,
        payload_dir=payload_dir,
        observer_authority_sha256="a" * 64,
        token_protocol_sha256=protocol_sha256,
        patch_tokens_per_image=4,
        load_adapter=load,
    )

    assert loads == 1
    assert len(adapter.prepared) == len(adapter.generated) == 256
    assert adapter.generated == [
        (row.generation_seed, 1.0, 1.0, 192) for row in capability
    ]
    assert adapter.prepared[0][0] == ((0, 7, 11), (1, 7, 11))
    assert adapter.prepared[8][0] == ((3, 7, 11), (2, 7, 11))
    assert all(row[2:] == ((0, 1), 4) for row in adapter.prepared)
    completion = validate_prism_completion_bundle_bytes(output.read_bytes())
    assert completion.phase == "diagnostic" and len(completion.rows) == 256
    assert tuple((row.pair_handle, row.channel) for row in completion.rows) == tuple(
        (row.pair_handle, row.channel) for row in capability
    )
    progress_value = json.loads(progress.read_bytes())
    assert progress_value["completed_rows"] == 256
    assert progress_value["last_pair_handle"] == capability[-1].pair_handle
    assert progress_value["last_channel"] == capability[-1].channel


def test_observer_refuses_overwrite_before_loading_model(tmp_path: Path) -> None:
    output = tmp_path / "completion.json"
    output.write_text("occupied")
    loads = 0

    def load() -> tuple[_Adapter, _Processor]:
        nonlocal loads
        loads += 1
        return _Adapter(), _Processor()

    try:
        observe_prism_cue_panel(
            output,
            tmp_path / "progress.json",
            phase="diagnostic",
            capability=(),
            prompt_bundle=_prompt_bundle(),
            payload_authorities=(),
            payload_dir=tmp_path,
            observer_authority_sha256="a" * 64,
            token_protocol_sha256="b" * 64,
            patch_tokens_per_image=4,
            load_adapter=load,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("observer accepted an existing output")
    assert loads == 0


@pytest.mark.parametrize("mutation", ("changed-payload", "unregistered-file", "duplicate-row"))
def test_observer_rejects_input_drift_before_loading_model(
    tmp_path: Path, mutation: str
) -> None:
    payload_dir = tmp_path / "payloads"
    authorities, digests = _payloads(payload_dir)
    capability = list(_capability(digests))
    if mutation == "changed-payload":
        (payload_dir / f"{digests[0]}.png").write_bytes(b"changed")
    elif mutation == "unregistered-file":
        (payload_dir / "extra.png").write_bytes(b"extra")
    else:
        capability[1] = capability[0]
    loads = 0

    def load() -> tuple[_Adapter, _Processor]:
        nonlocal loads
        loads += 1
        return _Adapter(), _Processor()

    with pytest.raises(ValueError):
        observe_prism_cue_panel(
            tmp_path / "completion.json",
            tmp_path / "progress.json",
            phase="diagnostic",
            capability=tuple(capability),
            prompt_bundle=_prompt_bundle(),
            payload_authorities=authorities,
            payload_dir=payload_dir,
            observer_authority_sha256="a" * 64,
            token_protocol_sha256="b" * 64,
            patch_tokens_per_image=4,
            load_adapter=load,
        )
    assert loads == 0


def test_observer_never_retries_invalid_completion_ids(tmp_path: Path) -> None:
    payload_dir = tmp_path / "payloads"
    authorities, digests = _payloads(payload_dir)
    capability = _capability(digests)
    prompts = _prompt_bundle()
    processor = _Processor()
    protocol = derive_prism_token_protocol(processor, prompts)
    protocol_sha256 = hashlib.sha256(canonical_json_bytes(asdict(protocol))).hexdigest()
    adapter = _Adapter()

    def invalid_generate(*_: object, **__: object) -> tuple[bool, ...]:
        return (True,)

    adapter.generate = invalid_generate  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="completion IDs"):
        observe_prism_cue_panel(
            tmp_path / "completion.json",
            tmp_path / "progress.json",
            phase="diagnostic",
            capability=capability,
            prompt_bundle=prompts,
            payload_authorities=authorities,
            payload_dir=payload_dir,
            observer_authority_sha256="a" * 64,
            token_protocol_sha256=protocol_sha256,
            patch_tokens_per_image=4,
            load_adapter=lambda: (adapter, processor),
        )
    assert len(adapter.prepared) == 1
    assert not (tmp_path / "completion.json").exists()


def _cli_args(tmp_path: Path) -> list[str]:
    return [
        "--phase",
        "diagnostic",
        "--capability",
        str(tmp_path / "capability.json"),
        "--prompt-bundle",
        str(tmp_path / "prompts.json"),
        "--payload-manifest",
        str(tmp_path / "payloads.json"),
        "--payload-dir",
        str(tmp_path / "payloads"),
        "--observer-authority-sha256",
        "a" * 64,
        "--token-protocol-sha256",
        "b" * 64,
        "--model-root",
        str(tmp_path / "model"),
        "--snapshot-manifest",
        str(tmp_path / "snapshot.json"),
        "--fixture",
        str(tmp_path / "fixture.json"),
        "--patch-tokens-per-image",
        "4",
        "--output",
        str(tmp_path / "completion.json"),
        "--progress-output",
        str(tmp_path / "progress.json"),
        "--execute-observer",
    ]


def test_observer_cli_accepts_only_anonymous_local_capabilities(tmp_path: Path) -> None:
    parsed = parse_args(_cli_args(tmp_path))

    assert parsed.phase == "diagnostic"
    assert parsed.execute_observer is True
    for forbidden in (
        "--scoring",
        "--truth",
        "--label",
        "--class-name",
        "--dataset-root",
        "--clean",
        "--test",
        "--url",
    ):
        with pytest.raises(SystemExit):
            parse_args([*_cli_args(tmp_path), forbidden, "forbidden"])


def test_observer_cli_requires_explicit_execution_flag(tmp_path: Path) -> None:
    args = _cli_args(tmp_path)
    args.remove("--execute-observer")

    with pytest.raises(SystemExit):
        parse_args(args)


def test_observer_main_authenticates_local_files_before_fake_model_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload_dir = tmp_path / "payloads"
    authorities, digests = _payloads(payload_dir)
    capability = _capability(digests)
    prompts = _prompt_bundle()
    processor = _Processor()
    protocol = derive_prism_token_protocol(processor, prompts)
    protocol_sha256 = hashlib.sha256(canonical_json_bytes(asdict(protocol))).hexdigest()
    (tmp_path / "capability.json").write_bytes(
        canonical_json_bytes(
            {
                "rows": [asdict(row) for row in capability],
                "schema": "sfora-prism-diagnostic-capability-v1",
            }
        )
    )
    (tmp_path / "prompts.json").write_bytes(
        canonical_json_bytes(asdict(prompts))
    )
    (tmp_path / "payloads.json").write_bytes(
        canonical_json_bytes(
            {
                "payloads": [asdict(row) for row in authorities],
                "schema": "sfora-prism-payload-manifest-v1",
            }
        )
    )
    for name in ("snapshot.json", "fixture.json"):
        (tmp_path / name).write_text("sealed")
    model = tmp_path / "model"
    model.mkdir()
    adapter = _Adapter()
    model_loads = 0

    def load_real_adapter(
        model_root: Path, snapshot_manifest: Path, fixture_path: Path
    ) -> tuple[_Adapter, _Processor, int]:
        nonlocal model_loads
        assert model_root == model
        assert snapshot_manifest == tmp_path / "snapshot.json"
        assert fixture_path == tmp_path / "fixture.json"
        model_loads += 1
        return adapter, processor, 4

    monkeypatch.setattr(observer_cli, "_load_real_adapter", load_real_adapter)
    args = _cli_args(tmp_path)
    token_index = args.index("--token-protocol-sha256") + 1
    args[token_index] = protocol_sha256

    assert observer_cli.main(args) == 0
    assert model_loads == 1
    completion = validate_prism_completion_bundle_bytes(
        (tmp_path / "completion.json").read_bytes()
    )
    assert len(completion.rows) == 256


def test_observer_rejects_symlinked_prompt_bundle(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(canonical_json_bytes(asdict(_prompt_bundle())))
    link = tmp_path / "prompts.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="path"):
        observer_cli._load_prompt_bundle(link)
