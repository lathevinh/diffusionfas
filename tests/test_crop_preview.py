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

from diffusionfas.celebahq_geometry import REAL_PIXEL_POLICY, crop_celebahq, crop_real_pixels, square_geometry
from diffusionfas.crop_preview import (
    AUTHORITY, BUDGET, SOURCES, cache_samples, checked_inputs, read_native_frames,
    run_preview, select_videos, summarize_preview, real_pixel_selection, AUDIT_PLAN_SHA,
)
from diffusionfas.data import Video
from diffusionfas.io import digest, object_digest, read_json, write_json
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

    def test_real_pixel_renderer_matches_in_bounds_reference_without_padding(self):
        rng = np.random.default_rng(42)
        rgb = rng.integers(0, 256, (320, 320, 3), dtype=np.uint8)
        for angle in (0., .3, -.7):
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            points = (POINTS - [64, 49]) @ rotation.T + [160, 160]
            actual, geometry = crop_real_pixels(rgb, points)
            reference, _ = crop_celebahq(rgb, points)
            np.testing.assert_array_equal(actual, reference)
            self.assertEqual(geometry["policy"], REAL_PIXEL_POLICY["name"])
            self.assertEqual(geometry["padding_fraction"], 0)
            self.assertTrue(geometry["bilinear_support_verified"])
            self.assertGreater(geometry["minimum_source_margin_pixels"], 0)

    def test_real_pixel_renderer_rejects_outside_before_rasterization(self):
        rgb = np.zeros((96, 128, 3), dtype=np.uint8)
        with patch("diffusionfas.celebahq_geometry.cv2.warpAffine") as warp:
            with self.assertRaisesRegex(MediaFailure, "square_outside"):
                crop_real_pixels(rgb, POINTS)
            warp.assert_not_called()
        with self.assertRaises(ValueError):
            crop_real_pixels(rgb.astype(np.float32), POINTS)

    def test_real_pixel_renderer_closed_bounds_without_epsilon(self):
        points = np.array([[30., 40.], [50., 40.], [40., 50.], [35., 60.], [45., 60.]])
        rgb = np.full((96, 128, 3), 127, dtype=np.uint8)
        result, meta = crop_real_pixels(rgb, points)
        self.assertEqual(meta["minimum_source_margin_pixels"], 0)
        self.assertTrue(np.all(result == 127))
        points[:, 0] -= 1e-9
        with self.assertRaisesRegex(MediaFailure, "square_outside"):
            crop_real_pixels(rgb, points)

    def test_real_pixel_renderer_checks_actual_interpolation_support(self):
        rgb = np.zeros((320, 320, 3), dtype=np.uint8)
        mask = np.ones((1024, 1024), dtype=np.float32)
        mask[0, 0] = .999
        with patch("diffusionfas.celebahq_geometry.cv2.warpAffine", return_value=mask) as warp:
            with self.assertRaisesRegex(MediaFailure, "bilinear_support"):
                crop_real_pixels(rgb, POINTS + [100, 100])
            self.assertEqual(warp.call_count, 1)

    def test_msu_selection_keeps_fixed_middle_gate_and_original_exclusions(self):
        rows = [replace(video("MSU-MFSD", label, f"{partition}-{label}-{i}"), partition=partition,
                        cache_path="/unused/cache.npz", cache_sha256="cache")
                for partition, label, count in (("fitting", "bona_fide", 18),
                                                 ("validation", "bona_fide", 6),
                                                 ("validation", "attack", 18))
                for i in range(count)]
        rows[-1] = replace(rows[-1], status="preparation_failure:decode")
        records = [{"key": v.key, "dataset": v.dataset, "partition": v.partition, "label": v.label,
                    "status": v.status, "geometry": {"all_cached_frames_fit": True,
                    "middle_T16_fit": v != rows[-2], "has_any_valid_T16_window": True,
                    "cache_sha256": v.cache_sha256} if v != rows[-1] else None} for v in rows]
        audit = {"bonafide_manifest": "/unused/bona.json", "diagnostic_manifest": "/unused/val.json",
                 "bonafide_manifest_sha256": "bona", "diagnostic_manifest_sha256": "val",
                 "population_sha256": object_digest([v.to_dict() for v in rows])}
        documents = {"plan.json": audit, "private-results.json": {"plan_sha256": AUDIT_PLAN_SHA,
                     "records": records}, "supervision.json": {"completed": True,
                     "plan_sha256": AUDIT_PLAN_SHA}, "bona.json": {}, "val.json": {}}
        validation = [v for v in rows if v.partition == "validation"]
        with patch("diffusionfas.crop_preview.read_json",
                   side_effect=lambda path, *args: documents[path.name]), \
                patch("diffusionfas.crop_fit_audit.population", return_value=rows):
            selected, cohort = real_pixel_selection(validation, Path("/unused"))
            reversed_selected, _ = real_pixel_selection(list(reversed(validation)), Path("/unused"))
            self.assertEqual(selected, reversed_selected)
            self.assertEqual(len(selected), 6)
            self.assertEqual(Counter(v.label for v in selected), {"bona_fide": 3, "attack": 3})
            self.assertTrue(all(v.dataset == "MSU-MFSD" and v.partition == "validation" for v in selected))
            self.assertEqual(len(cohort["eligible_keys"]), 40)
            self.assertEqual([e["key"] for e in cohort["excluded"]], [rows[-2].key, rows[-1].key])
            self.assertFalse(cohort["model_scoring_authorized"])
            with self.assertRaisesRegex(ValueError, "differs from audited"):
                real_pixel_selection(validation[:-1], Path("/unused"))
            records[-2]["geometry"]["middle_T16_fit"] = True
            with self.assertRaisesRegex(ValueError, "cohort counts changed"):
                real_pixel_selection(validation, Path("/unused"))

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

    def execute_worker(self, root, fail_first=False, real_pixels=False):
        media, frames, template = self.make_media(root)
        cache = root / "cache.npz"
        np.savez(cache, frames=np.stack([cv2.resize(f, (256, 256), interpolation=cv2.INTER_CUBIC)
                                        for f in frames]),
                 timestamps=template.timestamps, native_timestamps=template.timestamps,
                 native_orders=np.arange(17), boxes=np.tile([0., 0., 128., 96.], (17, 1)),
                 landmarks=np.tile(POINTS * .3 + [45, 25] if real_pixels else POINTS, (17, 1, 1)))
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
        for domain in (("MSU-MFSD",) if real_pixels else SOURCES):
            for label in ("bona_fide", "attack"):
                for index in range(3 if real_pixels else 2):
                    row = replace(template, key=f"{domain}/{label}-{index}", dataset=domain, label=label,
                                  group=f"{label}-{index}", cache_path=str(cache), cache_sha256=digest(cache))
                    entries.append({"video": row.to_dict(), "member": "fixture.avi",
                                    "samples": cache_samples(row), "receipt_sha256": "synthetic"})
        plan = {"selected": entries, "fas_root": str(fas_root),
                "preprocessing_sha256": digest(preprocessing), "msu_reader_sha256": digest(reader),
                "decoder_config": {}, "source_manifest": "unused.json",
                "pool_originals": 12, "pool_prepared": 12}
        if real_pixels:
            plan["real_pixel_cohort"] = {"test": True}
            plan["policy"] = REAL_PIXEL_POLICY
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
            if real_pixels:
                raise AssertionError("reflected renderer forbidden in real-pixel preview")
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
            self.assertEqual(stager.call_count, 6 if real_pixels else 12)
            self.assertEqual(native.return_value.bind.call_count, 6 if real_pixels else 4)
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

    def test_real_pixel_worker_produces_six_video_review_without_reflected_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            report, images, output = self.execute_worker(Path(directory), real_pixels=True)
            self.assertTrue(report["completed"])
            self.assertEqual(report["counts"], {"rendered": 6, "frames": 18})
            self.assertEqual(len(images), 54)
            self.assertEqual(report["model_scoring_runs"], 0)
            self.assertEqual(report["policy"], REAL_PIXEL_POLICY)
            page = (output / "review.html").read_text()
            self.assertIn("6/6 videos rendered; 18/18 frames", page)
            self.assertIn("real source pixels only", page)
            self.assertNotIn("reflected boundaries", page)
            for path in (output / "records").glob("result-*.json"):
                for frame in read_json(path)["frames"]:
                    self.assertTrue(frame["geometry"]["bilinear_support_verified"])
                    self.assertEqual(frame["geometry"]["padding_fraction"], 0)
