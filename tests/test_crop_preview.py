from __future__ import annotations

import base64
from collections import Counter
from dataclasses import replace
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from diffusionfas.celebahq_geometry import crop_celebahq, square_geometry
from diffusionfas.crop_preview import (
    AUTHORITY, BUDGET, SOURCES, cache_samples, checked_inputs, read_native_frames,
    run_preview, select_videos, summarize_preview,
)
from diffusionfas.data import Video
from diffusionfas.io import digest, read_json, write_json
from diffusionfas.preprocessing import MediaFailure

POINTS = np.array([[40., 30.], [88., 30.], [64., 48.], [48., 68.], [80., 68.]])


def video(domain="CASIA-FASD", label="bona_fide", key="sample"):
    return Video(f"{domain}/{key}", domain, key, key, "validation", label,
                 tuple(i / 8 for i in range(17)), (), None, None, "prepared")


class Images(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.images.append(dict(attrs)["src"])


class CropPreviewTests(unittest.TestCase):
    def test_square_geometry_is_isotropic_and_rotation_equivariant(self):
        for angle in (0., 0.2, -0.5):
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            points = POINTS @ rotation.T + [100, 100]
            geometry = square_geometry(points)
            matrix = np.asarray(geometry["source_to_output_matrix"])
            linear = matrix[:, :2]
            np.testing.assert_allclose(linear @ linear.T, np.eye(2) * np.linalg.det(linear), atol=1e-12)
            np.testing.assert_allclose(matrix[:, :2] @ geometry["center"] + matrix[:, 2], [127.5] * 2)
            self.assertLessEqual(geometry["eye_distance_output_pixels"], 64.)
            quad = np.asarray(geometry["quad"])
            lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            np.testing.assert_allclose(lengths, geometry["side_source_pixels"])

    def test_invalid_landmarks_fail_explicitly(self):
        for points in (np.zeros((5, 2)), np.full((5, 2), np.nan), np.zeros((4, 2))):
            with self.assertRaises(MediaFailure):
                square_geometry(points)
        with self.assertRaises(ValueError):
            crop_celebahq(np.zeros((10, 10, 3), dtype=np.float32), POINTS)

    def test_reflection_preserves_solid_rgb_and_reports_padding(self):
        rgb = np.broadcast_to(np.array([40, 110, 220], dtype=np.uint8), (96, 128, 3)).copy()
        cropped, geometry = crop_celebahq(rgb, POINTS)
        self.assertEqual(cropped.shape, (256, 256, 3))
        np.testing.assert_array_equal(cropped, np.broadcast_to(rgb[0, 0], cropped.shape))
        self.assertGreater(geometry["padding_fraction"], 0)
        self.assertLess(geometry["padding_fraction"], 1)

    def test_sampling_is_seeded_stratified_and_excludes_other_roles(self):
        rows = [video(domain, label, f"{label}-{i}") for domain in SOURCES
                for label in ("bona_fide", "attack") for i in range(5)]
        rows += [replace(video(key="fitting"), partition="fitting"),
                 replace(video(key="calibration"), partition="calibration"),
                 replace(video(key="failed"), status="preparation_failure:missing_track_endpoint"),
                 replace(video(key="target"), dataset="OULU-NPU")]
        selected = select_videos(rows)
        self.assertEqual(selected, select_videos(list(reversed(rows))))
        self.assertEqual(len(selected), 12)
        self.assertEqual(set(Counter((v.dataset, v.label) for v in selected).values()), {2})
        self.assertTrue(all(v.partition == "validation" and v.dataset in SOURCES for v in selected))

    def test_cache_slots_match_lower_middle_integer_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            row = video()
            np.savez(path, timestamps=row.timestamps, native_timestamps=row.timestamps,
                     native_orders=np.arange(17), boxes=np.tile([0., 0., 128., 96.], (17, 1)),
                     landmarks=np.tile(POINTS, (17, 1, 1)))
            row = replace(row, cache_path=str(path), cache_sha256=digest(path))
            samples = cache_samples(row)
            self.assertEqual([s["cache_index"] for s in samples], [0, 7, 15])
            self.assertEqual([s["native_order"] for s in samples], [0, 7, 15])

    def test_changed_native_cache_metadata_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            row = video()
            np.savez(path, timestamps=row.timestamps, native_timestamps=np.array(row.timestamps) + 1,
                     native_orders=np.arange(17), boxes=np.tile([0., 0., 128., 96.], (17, 1)),
                     landmarks=np.tile(POINTS, (17, 1, 1)))
            with self.assertRaisesRegex(ValueError, "tracking metadata"):
                cache_samples(replace(row, cache_path=str(path), cache_sha256=digest(path)))

    def test_preview_authority_cannot_enable_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            write_json(path, {"authority": {**AUTHORITY, "model_scoring": True}, "budget": BUDGET})
            with self.assertRaises(PermissionError):
                checked_inputs(path, digest(path))

    def make_media(self, root):
        path = root / "synthetic.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (128, 96))
        self.assertTrue(writer.isOpened())
        try:
            yy, xx = np.indices((96, 128))
            for index in range(17):
                rgb = np.stack(((xx + index) % 256, (yy * 2) % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        frames = []
        capture = cv2.VideoCapture(str(path))
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        self.assertEqual(len(frames), 17)
        row = replace(video(), media_sha256=digest(path),
                      frame_hashes=tuple(hashlib.sha256(f.tobytes()).hexdigest() for f in frames))
        return path, frames, row

    def test_actual_synthetic_native_frames_are_verified_and_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, frames, row = self.make_media(Path(directory))
            selected = read_native_frames(row, path, {0, 7, 15}, time.monotonic() + 10)
            self.assertEqual(set(selected), {0, 7, 15})
            for index, rgb in selected.items():
                np.testing.assert_array_equal(rgb, frames[index])

    def test_mismatch_in_nonselected_native_frame_is_not_waived(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, row = self.make_media(Path(directory))
            bad = replace(row, frame_hashes=("0" * 64, *row.frame_hashes[1:]))
            with self.assertRaisesRegex(MediaFailure, "RGB_mismatch"):
                read_native_frames(bad, path, {7, 15}, time.monotonic() + 10)

    def execute_worker(self, root, fail_first=False):
        media, frames, template = self.make_media(root)
        cache = root / "cache.npz"
        np.savez(cache, frames=np.stack([cv2.resize(f, (256, 256), interpolation=cv2.INTER_CUBIC)
                                        for f in frames]),
                 timestamps=template.timestamps, native_timestamps=template.timestamps,
                 native_orders=np.arange(17), boxes=np.tile([0., 0., 128., 96.], (17, 1)),
                 landmarks=np.tile(POINTS, (17, 1, 1)))
        fas_root = root / "fas"
        preprocessing = fas_root / "src/fas/preprocessing.py"
        preprocessing.parent.mkdir(parents=True)
        preprocessing.write_text("# synthetic test commitment\n")
        reader = fas_root / "scripts/inventory_msu.py"
        reader.parent.mkdir()
        reader.write_text("# synthetic reader commitment\n")
        output = root / "preview"
        output.mkdir()
        entries = []
        for domain in SOURCES:
            for label in ("bona_fide", "attack"):
                for index in range(2):
                    row = replace(template, key=f"{domain}/{label}-{index}", dataset=domain, label=label,
                                  group=f"{label}-{index}", cache_path=str(cache), cache_sha256=digest(cache))
                    entries.append({"video": row.to_dict(), "member": "fixture.avi",
                                    "samples": cache_samples(row), "receipt_sha256": "synthetic"})
        plan = {"selected": entries, "fas_root": str(fas_root),
                "preprocessing_sha256": digest(preprocessing), "msu_reader_sha256": digest(reader),
                "decoder_config": {}, "source_manifest": "unused.json",
                "pool_originals": 12, "pool_prepared": 12}
        path = output / "plan.json"
        write_json(path, plan)

        def legacy(rgb, box, landmarks):
            return rgb.copy(), {"crop_xyxy_half_open": [0, 0, 128, 96],
                                "alignment_matrix": [[1., 0., 0.], [0., 1., 0.]]}

        module = SimpleNamespace(__file__=str(preprocessing), shared_context_crop=legacy)

        def stage(entry, fas, data, destination, stack, context, deadline, **kwargs):
            self.assertTrue(kwargs["allow_validation_attacks"])
            shutil.copyfile(media, destination)

        def replay(source, consume, timeout):
            for index, frame in enumerate(frames):
                consume(index, frame)

        crop_calls = 0

        def corrected(rgb, landmarks):
            nonlocal crop_calls
            crop_calls += 1
            if fail_first and crop_calls == 1:
                raise MediaFailure("synthetic_geometry_failure")
            return crop_celebahq(rgb, landmarks)

        with patch("diffusionfas.crop_preview.checked_inputs", return_value=plan), \
                patch("diffusionfas.crop_preview.importlib",
                      SimpleNamespace(import_module=lambda name: module)), \
                patch("diffusionfas.crop_preview.decoder_policy",
                      return_value=({d: "fas_strict_ffmpeg" if d == "MSU-MFSD" else "opencv"
                                     for d in SOURCES}, "synthetic")), \
                patch("diffusionfas.crop_preview.FasNativeReplay") as native, \
                patch("diffusionfas.crop_preview.stage_member", side_effect=stage) as stager, \
                patch("diffusionfas.crop_preview.crop_celebahq", side_effect=corrected):
            native.return_value.bind.return_value = replay
            run_preview(path, digest(path))
            self.assertEqual(stager.call_count, 12)
            self.assertEqual(native.return_value.bind.call_count, 4)
            with self.assertRaisesRegex(ValueError, "already attempted"):
                run_preview(path, digest(path))
        self.assertEqual(list((output / "staging").iterdir()), [])
        report = summarize_preview(path, {"completed": True})
        parser = Images()
        parser.feed((output / "review.html").read_text())
        for source in parser.images:
            self.assertTrue(source.startswith("data:image/png;base64,"))
            pixels = cv2.imdecode(np.frombuffer(base64.b64decode(source.split(",", 1)[1]),
                                                dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(pixels)
        return report, parser.images, output

    def test_worker_generates_self_contained_html_and_stops_for_human_review(self):
        with tempfile.TemporaryDirectory() as directory:
            report, images, output = self.execute_worker(Path(directory))
            self.assertTrue(report["completed"])
            self.assertEqual(report["human_review"], "pending")
            self.assertFalse(report["automatic_continuation"])
            self.assertEqual(report["model_scoring_runs"], 0)
            self.assertEqual(report["counts"], {"rendered": 12, "frames": 36})
            self.assertEqual(len(images), 108)
            self.assertEqual(read_json(output / "completion.json")["human_review"], "pending")
            page = (output / "review.html").read_text()
            self.assertIn("PAUSED FOR HUMAN REVIEW", page)
            self.assertNotIn("<script", page)
            self.assertNotIn(str(output), page)

    def test_failed_crop_is_visible_without_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            report, images, output = self.execute_worker(Path(directory), fail_first=True)
            self.assertEqual(report["counts"], {"rendered": 11, "technical_failure": 1, "frames": 33})
            self.assertEqual(len(images), 99)
            self.assertIn("synthetic_geometry_failure", (output / "review.html").read_text())
            self.assertEqual(report["sampling_replacements"], 0)
