from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
import hashlib
import importlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from diffusionfas.cli import manifest_payload, parser, run_prepare
from diffusionfas.data import Video
from diffusionfas.io import digest, read_json, scientific_config, object_digest, write_json
from diffusionfas.native_replay import FAS_NATIVE_HELPER_SHA256, FasNativeReplay, decoder_policy
from diffusionfas.preprocessing import prepare_video
from test_preprocessing import FixtureFaces


class Fixture:
    def __init__(self, root: Path, offset: int = 0) -> None:
        self.root = root
        self.extras: list[Video] = []
        self.source_root = root / "synthetic-fas"
        self.helper = self.source_root / "scripts/msu_replay282.py"
        self.helper.parent.mkdir(parents=True)
        self.helper.write_text("# synthetic native helper fixture\n")
        self.media = root / "synthetic.avi"
        writer = cv2.VideoWriter(str(self.media), cv2.VideoWriter_fourcc(*"FFV1"), 16, (32, 32))
        if not writer.isOpened():
            raise RuntimeError("synthetic FFV1 writer unavailable")
        for index in range(48):
            writer.write(np.full((32, 32, 3), index * 3 + offset, np.uint8))
        writer.release()
        capture = cv2.VideoCapture(str(self.media))
        hashes = []
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                hashes.append(hashlib.sha256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()).hexdigest())
        finally:
            capture.release()
        sequence = hashlib.sha256()
        for frame_hash in hashes:
            sequence.update(f"32:32:3072:{frame_hash}\n".encode())
        self.video = Video(
            key="MSU-MFSD/synthetic", dataset="MSU-MFSD", group="synthetic", subject="synthetic",
            partition="fitting", label="bona_fide", timestamps=tuple(i / 16 for i in range(48)),
            frame_hashes=tuple(hashes), media_sha256=digest(self.media),
            decoded_sha256=sequence.hexdigest(), status="eligible_pending_tracking", media_path=str(self.media),
        )
        self.receipt = {
            "video_id": "synthetic", "binary_label": "bona_fide", "media_sha256": self.video.media_sha256,
            "byte_size": self.media.stat().st_size, "media_relpath": "release/synthetic.avi",
            "decode_status": "decoded", "frame_count": 48,
            "decoded_rgb_sequence_sha256": self.video.decoded_sha256,
            "frame_index": [
                {"decode_order": i, "timestamp_seconds": str(Fraction(i, 16)), "rgb_sha256": value,
                 "width": 32, "height": 32, "decode_success": True}
                for i, value in enumerate(hashes)
            ],
        }
        write_json(root / "media/0.json", self.receipt)
        write_json(root / "transactions/0.json", {
            "video_id": "synthetic", "binary_label": "bona_fide",
            "media_sha256": self.video.media_sha256, "media_record_sha256": digest(root / "media/0.json"),
        })
        write_json(root / "source.json", {"rows": [{
            "dataset": "MSU-MFSD", "video_id": "synthetic", "binary_label": "bona_fide",
            "record_path": str(root / "transactions/0.json"),
            "record_sha256": digest(root / "transactions/0.json"),
        }]})
        self.provenance = {"source_manifest_sha256": digest(root / "source.json")}
        self.config = {
            "target": "OULU-NPU", "execution": {"source_decode": True, "target_decode": False},
            "clip": {"fps": 8}, "budget": {"max_seconds": 30, "max_output_bytes": 64 * 1024**2},
            "preprocessing": {"decoders": {"MSU-MFSD": "fas_strict_ffmpeg"},
                              "native_replay_helper_sha256": digest(self.helper)},
        }

    def arguments(self, *, real_source_root: Path | None = None):
        if real_source_root is not None:
            self.config["preprocessing"]["native_replay_helper_sha256"] = FAS_NATIVE_HELPER_SHA256
        write_json(self.root / "config.json", self.config)
        write_json(self.root / "manifest.json", manifest_payload([self.video, *self.extras], self.provenance))
        return parser().parse_args([
            "prepare", "--config", str(self.root / "config.json"),
            "--manifest", str(self.root / "manifest.json"), "--partition", "fitting",
            "--cache-root", str(self.root / "cache"), "--out", str(self.root / "prepared.json"),
            "--fas-root", str(real_source_root or self.source_root),
            "--preprocessing-sha256", "synthetic-provider", "--detector", str(self.root / "unused.onnx"),
            "--detector-sha256", "synthetic-provider", "--source-manifest", str(self.root / "source.json"),
        ])

    def module(self, failure: Exception | None = None):
        def replay(path, receipt, consume, output, *, timeout):
            if failure is not None:
                write_json(output, {"strict_success": False, "failure": str(failure)})
                raise failure
            capture = cv2.VideoCapture(str(path))
            order = 0
            try:
                while True:
                    ok, bgr = capture.read()
                    if not ok:
                        break
                    consume(order, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    order += 1
            finally:
                capture.release()
            write_json(output, {"strict_success": True, "frames_verified": order})

        return SimpleNamespace(
            __file__=str(self.helper),
            decoded_tokens=lambda media: [(f["width"], f["height"], f["rgb_sha256"]) for f in media["frame_index"]],
            strict_replay=Mock(side_effect=replay),
        )


@contextmanager
def fake_runtime(fixture: Fixture, failure: Exception | None = None):
    module = fixture.module(failure)
    original_import = importlib.import_module

    def import_native(name, package=None):
        return module if name == "msu_replay282" else original_import(name, package)

    with patch.object(sys, "path", list(sys.path)), patch(
        "diffusionfas.native_replay.importlib.import_module", side_effect=import_native,
    ), patch("diffusionfas.preprocessing.FasFaceProvider", return_value=FixtureFaces()) as provider:
        yield module, provider


class NativeReplayTests(unittest.TestCase):
    def test_policy_default_compatibility_and_scientific_binding(self) -> None:
        legacy, helper = decoder_policy({})
        self.assertTrue(all(value == "opencv" for value in legacy.values()))
        self.assertIsNone(helper)
        config = read_json(Path(__file__).resolve().parents[1] / "configs/offline-v1.json")
        selected, helper = decoder_policy(config)
        self.assertEqual(selected["MSU-MFSD"], "fas_strict_ffmpeg")
        self.assertEqual(selected["CASIA-FASD"], "opencv")
        self.assertEqual(helper, FAS_NATIVE_HELPER_SHA256)
        changed = {**config, "preprocessing": {"decoders": {"MSU-MFSD": "opencv"}}}
        self.assertNotEqual(object_digest(scientific_config(config)), object_digest(scientific_config(changed)))

    def test_invalid_policy_never_silently_falls_back(self) -> None:
        for settings in (
            {"decoders": {"unknown": "opencv"}},
            {"decoders": {"MSU-MFSD": "typo"}},
            {"decoders": {"MSU-MFSD": "fas_strict_ffmpeg"}},
            {"decoders": {"OULU-NPU": "fas_strict_ffmpeg"}, "native_replay_helper_sha256": "a" * 64},
            {"decoder": "fas_strict_ffmpeg"},
            {"native_replay_helper_sha256": 42},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                decoder_policy({"preprocessing": settings})

    def test_cli_native_success_persists_policy_and_full_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            arguments = fixture.arguments()
            with fake_runtime(fixture) as (module, _):
                run_prepare(arguments)
            result = read_json(arguments.out)
            self.assertEqual(len(result["videos"]), 1)
            self.assertEqual(result["videos"][0]["status"], "prepared")
            self.assertEqual(module.strict_replay.call_count, 2)
            self.assertEqual(result["provenance"]["decoder_policy"]["source_manifest_sha256"],
                             fixture.provenance["source_manifest_sha256"])
            self.assertEqual(result["provenance"]["decoder_policy"]["native_replay_helper_sha256"], digest(fixture.helper))
            with np.load(result["videos"][0]["cache_path"], allow_pickle=False) as cache:
                self.assertEqual(cache["frames"].shape, (24, 256, 256, 3))
            diagnostics = list((arguments.cache_root / "native-replay").glob("*.json"))
            self.assertEqual(len(diagnostics), 2)
            self.assertTrue(all(read_json(p)["frames_verified"] == 48 for p in diagnostics))

    def test_cli_legacy_opencv_never_loads_native_helper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.config.pop("preprocessing")
            arguments = fixture.arguments()
            arguments.source_manifest = None
            with fake_runtime(fixture) as (module, _):
                run_prepare(arguments)
            self.assertEqual(read_json(arguments.out)["videos"][0]["status"], "prepared")
            module.strict_replay.assert_not_called()
            self.assertFalse((arguments.cache_root / "native-replay").exists())

    def test_domain_dispatch_retains_historical_and_untouched_target_originals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            other = Fixture(Path(directory) / "other", offset=7)
            casia = replace(other.video, key="CASIA-FASD/synthetic", dataset="CASIA-FASD")
            historical = replace(
                casia, key="CASIA-FASD/historical", group="historical", subject="historical",
                status="historical_decode_failure", timestamps=(), frame_hashes=(),
                media_sha256=None, decoded_sha256=None, media_path=None,
            )
            target = replace(historical, key="OULU-NPU/untouched", dataset="OULU-NPU",
                             group="untouched", subject="untouched", partition="evaluation")
            fixture.extras = [casia, historical, target]
            arguments = fixture.arguments()
            with fake_runtime(fixture), patch(
                "diffusionfas.preprocessing.prepare_video", wraps=prepare_video,
            ) as prepare:
                run_prepare(arguments)
            dispatch = {call.args[0].dataset: call.kwargs["native_replay"] for call in prepare.call_args_list}
            self.assertEqual(len(prepare.call_args_list), 2)
            self.assertIsNotNone(dispatch["MSU-MFSD"])
            self.assertIsNone(dispatch["CASIA-FASD"])
            rows = {r["key"]: r for r in read_json(arguments.out)["videos"]}
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[historical.key], read_json(arguments.manifest)["videos"][2])
            self.assertEqual(rows[target.key], read_json(arguments.manifest)["videos"][3])

    def test_native_binding_enforces_selection_bytes_and_immutable_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            with fake_runtime(fixture):
                native = FasNativeReplay(fixture.source_root, digest(fixture.helper),
                                         fixture.root / "source.json", digest(fixture.root / "source.json"))
                prefix = fixture.root / "native"
                with self.assertRaisesRegex(ValueError, "committed selection"):
                    native.bind(fixture.video, prefix, expected_receipt_sha256="0" * 64)
                wrong = fixture.root / "other.avi"
                wrong.write_bytes(b"not-the-bound-original")
                with self.assertRaisesRegex(ValueError, "media bytes"):
                    native.bind(fixture.video, prefix)(wrong, lambda index, rgb: None, 5)
                native.bind(fixture.video, prefix)(fixture.media, lambda index, rgb: None, 5)
                with self.assertRaisesRegex(ValueError, "new native replay receipt"):
                    native.bind(fixture.video, prefix)(fixture.media, lambda index, rgb: None, 5)

    def test_permission_and_missing_manifest_fail_before_detector_or_decode(self) -> None:
        for operation in ("source", "target", "missing-manifest"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                if operation == "source":
                    fixture.config["execution"]["source_decode"] = False
                arguments = fixture.arguments()
                if operation == "target":
                    arguments.partition = "evaluation"
                if operation == "missing-manifest":
                    arguments.source_manifest = None
                error = ValueError if operation == "missing-manifest" else PermissionError
                with fake_runtime(fixture) as (module, provider), self.assertRaises(error):
                    run_prepare(arguments)
                provider.assert_not_called()
                module.strict_replay.assert_not_called()

    def test_helper_manifest_and_imported_frame_bindings_fail_closed(self) -> None:
        for mismatch in ("helper", "manifest", "frames", "receipt"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                if mismatch == "helper":
                    fixture.config["preprocessing"]["native_replay_helper_sha256"] = "0" * 64
                elif mismatch == "manifest":
                    fixture.provenance["source_manifest_sha256"] = "0" * 64
                elif mismatch == "frames":
                    fixture.video = replace(fixture.video, frame_hashes=("0" * 64, *fixture.video.frame_hashes[1:]))
                else:
                    with (fixture.root / "media/0.json").open("a") as stream:
                        stream.write(" ")
                arguments = fixture.arguments()
                with fake_runtime(fixture) as (module, provider), self.assertRaises(ValueError):
                    run_prepare(arguments)
                provider.assert_not_called()
                module.strict_replay.assert_not_called()

    def test_native_failures_retain_original_and_never_opencv_fallback(self) -> None:
        for failure, status in (
            (ValueError("RGB identity/count mismatch"), "native_ffmpeg_replay_failure"),
            (ValueError("extra RGB bytes"), "native_ffmpeg_replay_failure"),
            (ValueError("strict decoder returncode=1 stderr_bytes=1; retained diagnostic"), "native_ffmpeg_replay_failure"),
            (TimeoutError("absolute decoder deadline"), "native_ffmpeg_deadline"),
        ):
            with self.subTest(failure=str(failure)), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                arguments = fixture.arguments()
                with fake_runtime(fixture, failure), patch(
                    "diffusionfas.preprocessing.cv2.VideoCapture", side_effect=AssertionError("unexpected fallback"),
                ):
                    run_prepare(arguments)
                result = read_json(arguments.out)
                self.assertEqual(len(result["videos"]), 1)
                self.assertEqual(result["videos"][0]["status"], f"preparation_failure:{status}")
                self.assertIsNone(result["videos"][0]["cache_path"])
                self.assertTrue(list((arguments.cache_root / "native-replay").glob("*.json")))

    def test_unknown_native_errors_are_not_classified_as_success_or_media_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            arguments = fixture.arguments()
            with fake_runtime(fixture, ValueError("unexpected native programming error")), self.assertRaisesRegex(
                ValueError, "unexpected native programming error",
            ):
                run_prepare(arguments)
            self.assertFalse(arguments.out.exists())

    def test_native_diagnostic_bytes_are_included_in_output_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.config["budget"]["max_output_bytes"] = 1
            arguments = fixture.arguments()
            with fake_runtime(fixture), patch("diffusionfas.preprocessing.prepare_video") as prepare:
                def oversized(*args, **kwargs):
                    path = arguments.cache_root / "native-replay/failure.stderr"
                    path.parent.mkdir(parents=True)
                    path.write_bytes(b"diagnostic-over-budget")
                    return fixture.video

                prepare.side_effect = oversized
                with self.assertRaisesRegex(OSError, "including native diagnostics"):
                    run_prepare(arguments)
            self.assertFalse(arguments.out.exists())

    @unittest.skipUnless(os.environ.get("DIFFUSIONFAS_FAS_ROOT"), "requires explicit --fas-root for native integration")
    def test_actual_pinned_FAS_ffmpeg_through_cli_on_synthetic_video(self) -> None:
        source_root = Path(os.environ["DIFFUSIONFAS_FAS_ROOT"])
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                if corrupt:
                    fixture.video = replace(fixture.video, frame_hashes=("0" * 64, *fixture.video.frame_hashes[1:]))
                    fixture.receipt["frame_index"][0]["rgb_sha256"] = "0" * 64
                    write_json(fixture.root / "media/corrupt.json", fixture.receipt)
                    write_json(fixture.root / "transactions/corrupt.json", {
                        "video_id": "synthetic", "binary_label": "bona_fide",
                        "media_sha256": fixture.video.media_sha256,
                        "media_record_sha256": digest(fixture.root / "media/corrupt.json"),
                    })
                    write_json(fixture.root / "corrupt-source.json", {"rows": [{
                        "dataset": "MSU-MFSD", "video_id": "synthetic", "binary_label": "bona_fide",
                        "record_path": str(fixture.root / "transactions/corrupt.json"),
                        "record_sha256": digest(fixture.root / "transactions/corrupt.json"),
                    }]})
                    fixture.provenance["source_manifest_sha256"] = digest(fixture.root / "corrupt-source.json")
                arguments = fixture.arguments(real_source_root=source_root)
                if corrupt:
                    arguments.source_manifest = fixture.root / "corrupt-source.json"
                with patch.object(sys, "path", list(sys.path)), patch(
                    "diffusionfas.preprocessing.FasFaceProvider", return_value=FixtureFaces(),
                ), patch("diffusionfas.preprocessing.cv2.VideoCapture", side_effect=AssertionError("unexpected OpenCV decode")):
                    run_prepare(arguments)
                result = read_json(arguments.out)
                expected = "preparation_failure:native_ffmpeg_replay_failure" if corrupt else "prepared"
                self.assertEqual(result["videos"][0]["status"], expected)
                checks = [read_json(p) for p in (arguments.cache_root / "native-replay").glob("*.json")]
                self.assertEqual(len(checks), 1 if corrupt else 2)
                self.assertTrue(all(r["strict_success"] == (not corrupt) for r in checks))
                if not corrupt:
                    self.assertTrue(all(r["frames_verified"] == 48 for r in checks))
                self.assertTrue(all(r["frame_substitution"] is False for r in checks))
