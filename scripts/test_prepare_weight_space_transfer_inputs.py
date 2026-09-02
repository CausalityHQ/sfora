import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from PIL import Image

from scripts.prepare_weight_space_transfer_inputs import (
    parse_arguments,
    prepare_burned_artifact,
)
from scripts.run_siglip_proxy_control import (
    ControlExampleBands,
    control_manifest_artifact_bytes,
)
from sfora.data import ImageExample


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (3, 2), color)


def _bands(*, burned_image: object | None = None) -> ControlExampleBands:
    examples = (
        ImageExample("cars/opt", _image((1, 2, 3)), 0),
        ImageExample("cars/clean", _image((4, 5, 6)), 49),
        ImageExample("cars/burned", burned_image or _image((7, 8, 9)), 82),
    )
    ordered = tuple(sorted(examples, key=lambda item: item.example_id))
    return ControlExampleBands(
        optimization=(examples[0],),
        clean_validation=(examples[1],),
        burned_diagnostic=(examples[2],),
        ordered_manifest=ordered,
    )


class BurnedArtifactTests(unittest.TestCase):
    def test_publishes_only_burned_content_addressed_pixels_deterministically(self) -> None:
        bands = _bands()
        source = control_manifest_artifact_bytes(bands)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first_raw = prepare_burned_artifact(
                bands=bands,
                control_manifest_raw=source,
                output=first,
            )
            second_raw = prepare_burned_artifact(
                bands=bands,
                control_manifest_raw=source,
                output=second,
            )

            self.assertEqual(first_raw, second_raw)
            self.assertTrue(first_raw.endswith(b"\n"))
            value = json.loads(first_raw)
            self.assertEqual(value["schema"], "sfora-weight-space-transfer-burned-input-v1")
            self.assertIs(value["claim_eligible"], False)
            self.assertEqual(len(value["examples"]), 1)
            row = value["examples"][0]
            self.assertEqual(row["example_id"], "cars/burned")
            self.assertEqual(row["label"], 82)
            self.assertEqual(row["source_ordinal"], 0)
            self.assertEqual(row["basename"], f"{row['image_sha256']}.png")
            payload = (first / "images" / row["basename"]).read_bytes()
            self.assertEqual(len(payload), row["byte_length"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), row["image_sha256"])
            self.assertEqual(payload, (second / "images" / row["basename"]).read_bytes())
            self.assertFalse(
                any("opt" in path.name or "clean" in path.name for path in first.rglob("*"))
            )

    def test_rejects_source_manifest_drift_before_publication(self) -> None:
        bands = _bands()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            with self.assertRaisesRegex(ValueError, "manifest"):
                prepare_burned_artifact(
                    bands=bands,
                    control_manifest_raw=b"{}\n",
                    output=output,
                )
            self.assertFalse(output.exists())

    def test_encoding_failure_removes_partial_namespace(self) -> None:
        class BrokenImage:
            def save(self, *_args: object, **_kwargs: object) -> None:
                raise RuntimeError("broken encoder")

        bands = _bands(burned_image=BrokenImage())
        source = control_manifest_artifact_bytes(bands)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            with self.assertRaisesRegex(RuntimeError, "broken encoder"):
                prepare_burned_artifact(
                    bands=bands,
                    control_manifest_raw=source,
                    output=output,
                )
            self.assertFalse(output.exists())

    def test_cli_is_explicit_and_has_no_model_or_clean_capability(self) -> None:
        required = [
            "--control-manifest",
            "/abs/control.json",
            "--control-manifest-sha256",
            "1" * 64,
            "--output",
            "/abs/output",
            "--execute-burned-preparation",
        ]
        parsed = parse_arguments(required)
        self.assertEqual(parsed.output, Path("/abs/output"))
        for flag in ("--checkpoint", "--model", "--clean-root", "--official-test"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(required + [flag, "/abs/forbidden"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_arguments(required[:-1])


if __name__ == "__main__":
    unittest.main()
