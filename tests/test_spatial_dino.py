from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from diffusionfas.gpu_probe import tensor_digest
from diffusionfas.io import object_digest
from diffusionfas.regional_dino import RegionSupportError
from diffusionfas.spatial_dino import (
    ARMS, DIMENSION, SEED, SETTINGS, PatchFusion, SourceModules, SpatialReference,
    control_permutation, fit, optimizer_for, paired_batch, patch_prior, points224,
    predict, prior_image, source_modules, validation_loss,
)
from diffusionfas.spatial_preparation import POLICY, PROPOSED_LIMITS, estimate_budget, fixture

FAS_ROOT = Path("/mnt/e/FAS")


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, DIMENSION)

    def forward_features(self, images):
        patches = F.adaptive_avg_pool2d(images, (16, 16)).flatten(2).transpose(1, 2)
        tokens = self.projection(patches)
        return {"x_norm_patchtokens": tokens, "x_norm_clstoken": tokens.mean(1)}


class TinyReference(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = TinyBackbone()
        self.global_head, self.patch_head = nn.Linear(DIMENSION, 2), nn.Linear(DIMENSION, 2)
        self.weights = None
        self.backbone.projection.register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        self.weights = output.detach().square().mean(-1).softmax(-1)

    def forward(self, images):
        self.weights = None
        features = self.backbone.forward_features(images)
        return (self.global_head(features["x_norm_clstoken"]),
                self.patch_head(F.normalize(features["x_norm_patchtokens"], dim=-1)*16), self.weights)


class SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source = source_modules(FAS_ROOT)
        cls.rows, cls.images, cls.priors = fixture()

    def loaders(self):
        return (lambda row: self.images[row["index"]].copy(),
                lambda row: self.priors[row["index"]].copy())

    def test_pixel_center_landmark_transfer_reuses_original_geometry(self):
        points = np.array([[76, 80], [148, 80], [112, 115], [86, 151], [138, 151]], dtype=float)
        transaction = {"landmarks": points.tolist(), "crop_shape": [224, 224, 3],
                       "geometry": {"alignment_matrix": [[1, 0, 0], [0, 1, 0]],
                                    "crop_xyxy_half_open": [0, 0, 224, 224], "context_scale": 1.3,
                                    "interpolation": "bicubic", "border": "constant_zero"}}
        np.testing.assert_allclose(points224(transaction), points, atol=1e-12)
        bad = copy.deepcopy(transaction)
        bad["landmarks"][0][0] = -20
        with self.assertRaises(RegionSupportError):
            points224(bad)
        bad = copy.deepcopy(transaction)
        bad["geometry"]["context_scale"] = 1.
        with self.assertRaises(ValueError):
            points224(bad)

    def test_soft_prior_has_positive_full_support_and_no_hard_crop_failure(self):
        points = np.array([[5, 5], [25, 5], [15, 15], [6, 25], [24, 25]], dtype=float)
        value = prior_image(points)
        self.assertEqual(value.shape, (224, 224))
        self.assertEqual(value.dtype, np.float32)
        self.assertTrue(np.all(value >= 1))
        self.assertTrue(np.all(value <= 2))
        self.assertGreater(float(value[5, 5]), float(value[200, 200]))
        value = patch_prior(value[None])
        self.assertEqual(value.shape, (1, 256))
        np.testing.assert_allclose(value.mean(1), 1, atol=1e-6)
        with self.assertRaises(RegionSupportError):
            prior_image(np.tile([50., 50.], (5, 1)))
        with self.assertRaises(ValueError):
            patch_prior(np.zeros((1, 224, 224), dtype=np.float32))

    def test_control_is_fixed_nonidentity_histogram_preserving_and_not_target_addressed(self):
        a = control_permutation()
        np.testing.assert_array_equal(a, control_permutation())
        np.testing.assert_array_equal(np.sort(a), np.arange(256))
        self.assertFalse(np.array_equal(a, np.arange(256)))
        priors = patch_prior(self.priors)
        np.testing.assert_array_equal(np.sort(priors, 1), np.sort(priors[:, a], 1))

    def test_fusion_initialization_matches_all_four_arms(self):
        hashes = []
        for arm in ARMS[1:]:
            torch.manual_seed(12)
            fusion = PatchFusion(arm)
            hashes.append(tensor_digest(fusion.state_dict()))
            self.assertTrue(torch.equal(fusion.local_head.weight, torch.zeros_like(fusion.local_head.weight)))
        self.assertEqual(len(set(hashes)), 1)

    def test_patch_weights_normalized_with_uniform_floor_and_adaptive_prior(self):
        tokens = torch.randn(6, 256, DIMENSION)
        priors = torch.from_numpy(patch_prior(self.priors))
        for arm in ARMS[1:]:
            fusion = PatchFusion(arm)
            weights = fusion.weights(tokens, priors)
            torch.testing.assert_close(weights.sum(1), torch.ones(6), atol=1e-6, rtol=0)
            self.assertTrue((weights >= SETTINGS["uniform_floor"]/256).all())
            if arm in ("uniform", "content"):
                torch.testing.assert_close(weights, torch.full_like(weights, 1/256), rtol=0, atol=0)
            else:
                self.assertGreater(float(weights.std()), 0)
        fusion = PatchFusion("anatomical")
        with self.assertRaises(ValueError):
            fusion.weights(tokens, torch.zeros_like(priors))
        with self.assertRaises(ValueError):
            fusion.weights(tokens[:, :255], priors)

    def test_gate_learns_through_global_loss_without_new_patch_labels(self):
        torch.manual_seed(1)
        fusion = PatchFusion("anatomical")
        optimizer = torch.optim.AdamW(fusion.parameters(), lr=.01)
        tokens = torch.randn(6, 256, DIMENSION, requires_grad=True)
        priors = torch.from_numpy(patch_prior(self.priors))
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(fusion(tokens, priors), torch.tensor([0, 1]*3)).backward()
            optimizer.step()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in fusion.gate.parameters()))
        self.assertGreater(float(fusion.gate[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(tokens.grad.abs().sum()), 0)

    def test_all_initial_logits_and_apl_match_reference_exactly(self):
        images = self.source.model.tensor_images(self.images[:2], "cpu")
        priors = torch.from_numpy(patch_prior(self.priors[:2]))
        torch.manual_seed(8)
        original = TinyReference().eval()
        expected = original(images)
        for arm in ARMS:
            torch.manual_seed(8)
            model = SpatialReference(TinyReference(), arm).eval()
            actual = model(images, priors)
            for a, b in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_optimizer_preserves_original_backbone_rates_and_decay(self):
        model = SpatialReference(TinyReference(), "reference")
        wrapped = {id(p): (group["lr"], group["weight_decay"])
                   for group in optimizer_for(model).param_groups for p in group["params"]}
        original = {id(p): (group["lr"], group["weight_decay"])
                    for group in self.source.model.optimizer_for(model.reference).param_groups
                    for p in group["params"]}
        self.assertEqual(wrapped, original)
        model = SpatialReference(TinyReference(), "uniform")
        frozen = {id(p) for p in model.fusion.gate.parameters()}
        self.assertFalse(any(id(p) in frozen for group in optimizer_for(model).param_groups
                             for p in group["params"]))

    def test_exact_pinned_augmentation_parity_including_redraws_and_PDA(self):
        load_rgb, load_prior = self.loaders()
        for step, fas, pda in ((0, 0., 0.), (1, .2, .2), (2, 0., 1.), (3, 1., 1.)):
            with self.subTest(step=step):
                original = self.source.augmentation.paired_batch(
                    self.rows, load_rgb, self.source.augmentation.FASAug(), SEED, 1, step,
                    fas_probability=fas, pda_probability=pda)
                spatial = paired_batch(
                    self.rows, load_rgb, load_prior, self.source.augmentation.FASAug(),
                    self.source, SEED, 1, step, fas_probability=fas, pda_probability=pda)
                for name in ("images", "labels", "domains", "patch_labels"):
                    np.testing.assert_array_equal(getattr(original, name), getattr(spatial, name))
                self.assertEqual(original.trace, {key: spatial.trace[key] for key in original.trace})
                again = paired_batch(
                    self.rows, load_rgb, load_prior, self.source.augmentation.FASAug(),
                    self.source, SEED, 1, step, fas_probability=fas, pda_probability=pda)
                self.assertEqual(spatial.trace, again.trace)

    def test_prior_follows_actual_flip_rotation_and_PDA_donor_slot(self):
        load_rgb, load_prior = self.loaders()
        batch = paired_batch(self.rows, load_rgb, load_prior, self.source.augmentation.FASAug(),
                             self.source, SEED, 1, 2, fas_probability=0., pda_probability=1.)
        fields = []
        for index, transform in zip(batch.trace["selected"], batch.trace["spatial_transforms"], strict=True):
            field = Image.fromarray(load_prior(self.rows[index]))
            if transform["flip"]:
                field = field.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            field = field.rotate(transform["angle"], Image.Resampling.BILINEAR, fillcolor=1.)
            fields.append(np.asarray(field).reshape(16, 14, 16, 14).mean((1, 3)).ravel())
        fields = np.stack(fields)
        self.assertTrue(any(slot >= 0 for slot in batch.trace["spatial_donor_slots"]))
        for i, donor in enumerate(batch.trace["spatial_donor_slots"]):
            expected = fields[i].copy()
            if donor >= 0:
                mask = batch.patch_labels[i] == 0
                expected[mask] = fields[donor][mask]
            expected /= expected.mean()
            np.testing.assert_allclose(batch.priors[i], expected, atol=2e-7, rtol=2e-7)

    def test_invalid_prior_does_not_silently_become_uniform(self):
        load_rgb, _ = self.loaders()
        with self.assertRaises(ValueError):
            paired_batch(self.rows, load_rgb, lambda row: np.zeros((224, 224), dtype=np.float32),
                         self.source.augmentation.FASAug(), self.source, SEED, 1, 0)
        with self.assertRaises(ValueError):
            paired_batch(self.rows[:2], load_rgb, lambda row: self.priors[0],
                         self.source.augmentation.FASAug(), self.source, SEED, 1, 0)

    def test_serialization_and_predict_replay_for_all_arms(self):
        load_rgb, load_prior = self.loaders()
        for arm in ARMS:
            torch.manual_seed(9)
            model = SpatialReference(TinyReference(), arm)
            expected = predict(model, self.rows, load_rgb, load_prior, self.source, "cpu", lambda: None)
            torch.manual_seed(10)
            restored = SpatialReference(TinyReference(), arm)
            restored.load_state_dict(model.state_dict(), strict=True)
            np.testing.assert_array_equal(
                predict(restored, self.rows, load_rgb, load_prior, self.source, "cpu", lambda: None), expected)

    def test_source_balanced_validation_and_missing_class_fail_explicitly(self):
        value = validation_loss(np.zeros(6), self.rows)
        self.assertAlmostEqual(value, np.log(2))
        with self.assertRaises(ValueError):
            validation_loss(np.zeros(5), self.rows[:5])
        with self.assertRaises(ValueError):
            validation_loss(np.full(6, np.nan), self.rows)

    def test_training_path_uses_earliest_source_checkpoint_and_full_patience(self):
        stub = ModuleType("synthetic_reference")
        stub.tensor_images, stub.losses = self.source.model.tensor_images, self.source.model.losses

        def check(model):
            for parameter in model.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError("invalid synthetic gradient")
        stub.check_gradients = check
        source = SourceModules(stub, self.source.augmentation)
        model = SpatialReference(TinyReference(), "anatomical")
        load_rgb, load_prior = self.loaders()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"trace.jsonl"
            with patch.dict(SETTINGS, {"epochs": 8, "patience": 2}), patch(
                "diffusionfas.spatial_dino.validation_loss", side_effect=[1., .5, .5, .5]
            ):
                saved = fit(model, self.rows, self.rows, load_rgb, load_prior, source, SEED,
                            "cpu", lambda: None, path)
                self.assertEqual(saved["settings_sha256"], object_digest(SETTINGS))
            self.assertEqual(saved["selected_epoch"], 2)
            self.assertEqual(saved["optimizer_steps"], 4)
            self.assertEqual(len(saved["history"]), 4)
            self.assertEqual(len(path.read_text().splitlines()), 4)
            self.assertEqual(tensor_digest(model.state_dict()), tensor_digest(saved["state_dict"]))

    def test_budget_uses_all_twenty_fits_and_never_promises_a_hard_upper_bound(self):
        probe = {"status": "complete", "scientific_training_starts": 0, "PAD_image_decodes": 0,
                 "results": [{"arm": arm, "measured_step_seconds": [.2, .3], "inference_seconds": .06,
                              "checkpoint_bytes": 100} for arm in ARMS]}
        inventory = [{"arm": arm, "maximum_optimizer_steps": 200, "validation_usable": 6}
                     for _ in range(4) for arm in ARMS]
        result = estimate_budget(inventory, probe)
        self.assertEqual(result["maximum_optimizer_steps"], 4000)
        self.assertEqual(result["estimated_checkpoint_bytes"], 2000)
        self.assertAlmostEqual(result["raw_max_epoch_synthetic_training_seconds"], 1200)
        self.assertFalse(result["is_guaranteed_upper_bound"])
        self.assertEqual(POLICY["scientific_fits_proposed"], 20)
        self.assertFalse(POLICY["automatic_expansion"])
        self.assertEqual(result["proposed_wall_cap_seconds"], PROPOSED_LIMITS["max_seconds"])
        self.assertEqual(PROPOSED_LIMITS, {"max_seconds": 432000, "max_rss_bytes": 8*1024**3,
                                        "max_gpu_bytes": 12*1024**3, "max_output_bytes": 16*1024**3})
        probe["scientific_training_starts"] = 1
        with self.assertRaises(ValueError):
            estimate_budget(inventory, probe)

    def test_cli_has_no_scientific_execution_or_approval_command(self):
        script = Path(__file__).resolve().parents[1]/"scripts/spatial_dino.py"
        for command in ("run", "approve"):
            result = subprocess.run([sys.executable, str(script), command], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid choice", result.stderr)


if __name__ == "__main__":
    unittest.main()
