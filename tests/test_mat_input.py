from __future__ import annotations

import base64
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from diffusionfas.celebahq_geometry import crop_real_pixels, square_geometry
from diffusionfas.io import digest, read_json, write_json
from diffusionfas.mat_input import POLICY, anatomical_masks, crop_native512, masked_error, model_input
from diffusionfas.mat_preview import (
    AUTHORITY, BUDGET, checked_inputs, configure_mat_gpu, reconstruct, reference_operator, render_review, summarize,
)
from diffusionfas.preprocessing import MediaFailure
from test_crop_preview import POINTS


class MaskedModel(torch.nn.Module):
    def forward(self, image, mask, z, label, truncation_psi, noise_mode):
        assert noise_mode == "const" and truncation_psi == 1
        assert label.shape == (1, 0)
        assert torch.all(image * (1 - mask) == 0)
        generated = torch.rand_like(image) * .5 + z[0, 0] * .1
        return image * mask + generated * (1 - mask)


class MATInputTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_native512_crop_is_same_square_and_not_upscaled_cached256(self):
        rng = np.random.default_rng(42)
        rgb = rng.integers(0, 256, (320, 320, 3), dtype=np.uint8)
        points = POINTS + [100, 100]
        result, meta = crop_native512(rgb, points)
        self.assertEqual(result.shape, (512, 512, 3))
        self.assertEqual(meta["padding_fraction"], 0)
        self.assertTrue(meta["bilinear_support_verified"])
        np.testing.assert_array_equal(meta["quad"], square_geometry(points)["quad"])
        old, _ = crop_real_pixels(rgb, points)
        self.assertFalse(np.array_equal(result, cv2.resize(old, (512, 512))))
        matrix = np.asarray(meta["source_to_output_matrix"])
        np.testing.assert_allclose(meta["output_landmarks"], points @ matrix[:, :2].T + matrix[:, 2])
        linear = matrix[:, :2]
        np.testing.assert_allclose(linear @ linear.T, np.eye(2) * np.linalg.det(linear), atol=1e-12)

    def test_crop_rejects_context_shortage_before_rasterization(self):
        with patch("diffusionfas.mat_input.cv2.warpAffine") as warp:
            with self.assertRaisesRegex(MediaFailure, "square_outside"):
                crop_native512(np.zeros((96, 128, 3), dtype=np.uint8), POINTS)
            warp.assert_not_called()

    def test_anatomical_mask_polarity_position_area_and_no_clipping(self):
        points = np.array([[192., 220.], [320., 220.], [256., 265.], [225., 308.], [287., 308.]])
        masks = anatomical_masks(points)
        self.assertEqual(list(masks), POLICY["mask_regions"])
        centers = [points[0], points[1], points[2], points[3:5].mean(axis=0)]
        for (name, known), center in zip(masks.items(), centers, strict=True):
            self.assertEqual(known.dtype, np.float32)
            self.assertEqual(known[int(center[1]), int(center[0])], 0)
            self.assertEqual(known[0, 0], 1)
            rx, ry = np.asarray(POLICY["ellipse_radii_in_eye_distances"][name]) * 128
            self.assertAlmostEqual(float((known == 0).sum()), float(np.pi * rx * ry), delta=30)
        points[0] = [1, 1]
        with self.assertRaisesRegex(MediaFailure, "mask_outside"):
            anatomical_masks(points)

    def test_normalization_and_missing_content_are_explicit(self):
        rgb = np.full((512, 512, 3), 255, dtype=np.uint8)
        known = np.ones((512, 512), dtype=np.float32)
        known[200:250, 200:250] = 0
        image, mask = model_input(rgb, known)
        self.assertEqual(image.shape, (1, 3, 512, 512))
        self.assertTrue(torch.all(image[mask.expand_as(image) == 0] == 0))
        self.assertTrue(torch.all(image[mask.expand_as(image) == 1] == 1))
        rgb[known == 1] = 0
        image, _ = model_input(rgb, known)
        self.assertTrue(torch.all(image[mask.expand_as(image) == 1] == -1))
        for bad in (known.astype(np.uint8), np.ones_like(known), np.zeros_like(known), known + .5):
            with self.assertRaises(ValueError):
                model_input(rgb, bad)

    def test_masked_error_ignores_visible_region_and_uses_actual_hole_denominator(self):
        rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        known = np.ones((512, 512), dtype=np.float32)
        known[100:120, 100:120] = 0
        predicted = np.zeros((512, 512, 3), dtype=np.float32)
        score, heat = masked_error(rgb, predicted, known)
        self.assertEqual(score, 1)
        self.assertTrue(np.all(heat[known == 1] == 0))
        predicted[known == 1] = 999
        other, _ = masked_error(rgb, predicted, known)
        self.assertEqual(score, other)

    def test_seed_controls_internal_randomness_and_original_hole_cannot_leak(self):
        rgb = np.full((512, 512, 3), 123, dtype=np.uint8)
        known = np.ones((512, 512), dtype=np.float32)
        known[180:240, 180:260] = 0
        model = MaskedModel().eval()
        state = torch.get_rng_state()
        first = reconstruct(model, rgb, known, 42, torch.device("cpu"))
        np.testing.assert_array_equal(torch.get_rng_state(), state)
        rgb[known == 0] = 255
        repeated = reconstruct(model, rgb, known, 42, torch.device("cpu"))
        np.testing.assert_array_equal(first, repeated)
        changed = reconstruct(model, rgb, known, 43, torch.device("cpu"))
        self.assertFalse(np.array_equal(first, changed))

    def test_authority_cannot_enable_AUROC_or_training(self):
        for field in ("AUROC", "training"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                write_json(path, {"authority": {**AUTHORITY, field: True}, "budget": BUDGET, "policy": POLICY})
                with self.assertRaises(PermissionError):
                    checked_inputs(path, digest(path))

    def test_reference_operator_cannot_be_overridden_by_nested_cuda_keyword(self):
        calls = []
        def operation(value, impl="cuda"):
            calls.append(impl)
            return value
        reference = reference_operator(operation)
        self.assertEqual(reference(42, impl="cuda"), 42)
        self.assertEqual(reference(43), 43)
        self.assertEqual(calls, ["ref", "ref"])

    def test_mat_explicitly_enables_deterministic_backend_without_environment_install(self):
        previous = torch.backends.cudnn.deterministic
        try:
            torch.backends.cudnn.deterministic = False
            with patch("diffusionfas.mat_preview.configure_gpu", return_value={"numeric": "FP32"}), \
                    patch("diffusionfas.mat_preview.torch.use_deterministic_algorithms") as deterministic:
                report = configure_mat_gpu({})
                deterministic.assert_called_once_with(True)
                self.assertTrue(torch.backends.cudnn.deterministic)
                self.assertTrue(report["deterministic_algorithms"])
                self.assertEqual(report["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        finally:
            torch.backends.cudnn.deterministic = previous

    def test_private_review_is_self_contained_and_binds_PNG_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "records").mkdir()
            (root / "images").mkdir()
            rgb = np.zeros((512, 512, 3), dtype=np.uint8)
            from diffusionfas.crop_preview import save_png
            image = save_png(root / "images/input.png", rgb)
            images = {kind: image for kind in ("masked", "restored", "residual")}
            row = {"index": 0, "failure": None, "frames": [{"clip_slot": 0, "original": image,
                    "masks": {"left_eye": {"hole_fraction": .01, "masked_mse": .1, "images": images}}}]}
            receipt = root / "records/result-00.json"
            write_json(receipt, row)
            plan = {"entries": [{"video": {"label": "bona_fide"}, "member": "fixture.avi"}]}
            path = root / "plan.json"
            write_json(path, plan)
            review = render_review(root, plan)
            self.assertEqual(review["rendered_frames"], 1)
            self.assertEqual(review["reconstructions"], 1)
            self.assertEqual(review["technical_failure_originals"], 0)
            class Images(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.images = []
                def handle_starttag(self, tag, attrs):
                    if tag == "img":
                        self.images.append(dict(attrs)["src"])
            parser = Images()
            parser.feed((root / "review.html").read_text())
            self.assertEqual(len(parser.images), 4)
            for source in parser.images:
                self.assertEqual(base64.b64decode(source.split(",", 1)[1]), (root / image["file"]).read_bytes())
            write_json(root / "completion.json", {"plan_sha256": digest(path), "completed": True,
                       "review": review, "records_sha256": {receipt.name: digest(receipt)}})
            report = summarize(path, {"completed": True})
            self.assertTrue(report["completed"])
            self.assertFalse(report["automatic_continuation"])
            self.assertNotIn("records_sha256", report["completion"])
            self.assertEqual(report["human_review"], "pending")
            (root / image["file"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "PNG binding"):
                render_review(root, plan)
