from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from diffusionfas.io import digest, write_json
from diffusionfas.models import ClipDenoiser, build_model, local_pretrained_model, scratch_model
from diffusionfas.scoring import residual_profile

torch.set_num_threads(1)

TINY = {"resolution": 16, "channels": [8, 16], "layers_per_block": 1,
        "train_timesteps": 20, "beta_schedule": "squaredcos_cap_v2"}


class ModelTests(unittest.TestCase):
    def test_scratch_backward_and_native_resolution(self) -> None:
        model = scratch_model(TINY, temporal=False)
        noisy = torch.randn(1, 3, 3, 16, 16)
        prediction = model(noisy, torch.tensor([2]))
        self.assertEqual(prediction.shape, noisy.shape)
        prediction.square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.backbone.parameters()))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 3, 32, 32), torch.tensor([2]))

    def test_frozen_backbone_zero_adapter_identity_and_learning(self) -> None:
        base = scratch_model(TINY, temporal=False)
        adapted = ClipDenoiser(copy.deepcopy(base.backbone), copy.deepcopy(base.scheduler),
                               temporal=True, freeze_backbone=True, adapter_channels=8)
        noisy, timestep = torch.randn(1, 3, 3, 16, 16), torch.tensor([2])
        self.assertTrue(torch.equal(base(noisy, timestep), adapted(noisy, timestep)))
        before = {name: p.clone() for name, p in adapted.backbone.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=0.01)
        adapted.train()
        self.assertFalse(adapted.backbone.training)
        loss = (adapted(noisy, timestep) - torch.randn_like(noisy)).square().mean()
        loss.backward()
        optimizer.step()
        self.assertTrue(all(p.grad is None for p in adapted.backbone.parameters()))
        self.assertTrue(all(torch.equal(p, before[n]) for n, p in adapted.backbone.named_parameters()))
        self.assertFalse(torch.equal(base(noisy, timestep), adapted(noisy, timestep)))

    def test_noise_score_deterministic_and_preserves_mode(self) -> None:
        model = scratch_model(TINY, temporal=True, adapter_channels=8)
        clips = torch.randn(2, 3, 3, 16, 16)
        kwargs = dict(alpha_bars=[0.9, 0.5, 0.1], draws=2, seed=3)
        first = residual_profile(model, clips, ["a", "b"], **kwargs)
        torch.manual_seed(999)
        second = residual_profile(model, clips, ["a", "b"], **kwargs)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, (2, 3))
        self.assertTrue(model.training)
        swapped = residual_profile(model, clips.flip(0), ["b", "a"], **kwargs)
        self.assertTrue(torch.equal(first.flip(0), swapped))

    def test_spatial_clip_score_equals_mean_frame_scores_with_addressed_noise(self) -> None:
        model = scratch_model(TINY, temporal=False).eval()
        clips = torch.randn(1, 3, 3, 16, 16)
        kwargs = dict(alpha_bars=[0.5], draws=1, seed=3)
        whole = residual_profile(model, clips, ["a"], **kwargs)
        pieces = [residual_profile(model, clips[:, i:i+1], ["a"], frame_addresses=[i], **kwargs)
                  for i in range(3)]
        self.assertTrue(torch.allclose(whole, torch.stack(pieces).mean(0), atol=1e-6))

    def test_local_safe_bundle_load_and_hash_rejection(self) -> None:
        base = scratch_model(TINY, temporal=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "config.json", dict(base.backbone.config))
            write_json(root / "scheduler_config.json", dict(base.scheduler.config))
            save_file(base.backbone.state_dict(), str(root / "model.safetensors"))
            manifest = {
                "upstream_revision": "synthetic-fixture-not-a-downloaded-model",
                "rights_reviewed": True,
                "files": {name: digest(root / name) for name in
                          ("config.json", "scheduler_config.json", "model.safetensors")},
            }
            write_json(root / "manifest.json", manifest)
            loaded = local_pretrained_model(root, digest(root / "manifest.json"),
                                            temporal=False, adapter_channels=8)
            noisy, timestep = torch.randn(1, 2, 3, 16, 16), torch.tensor([1])
            self.assertTrue(torch.equal(base(noisy, timestep), loaded(noisy, timestep)))
            with self.assertRaises(ValueError):
                local_pretrained_model(root, "bad-hash", temporal=True)

    def test_missing_pretrained_never_downloads(self) -> None:
        with self.assertRaisesRegex(ValueError, "no automatic download"):
            build_model({"model": {"variant": "pretrained_temporal", "local_bundle": None,
                                   "bundle_manifest_sha256": None, "adapter_channels": 8}})

    def test_learned_spatial_adapter_does_not_mix_frames(self) -> None:
        torch.manual_seed(5)
        base = scratch_model(TINY, temporal=False)
        model = ClipDenoiser(base.backbone, base.scheduler, temporal=True,
                            freeze_backbone=True, adapter_channels=8, temporal_kernel=1).eval()
        with torch.no_grad():
            model.adapter.output.weight.normal_(0, 0.1)
        clip = torch.randn(1, 3, 3, 16, 16)
        timestep = torch.tensor([2])
        together = model(clip, timestep)
        changed = clip.clone()
        changed[:, -1].add_(10)
        updated = model(changed, timestep)
        self.assertTrue(torch.equal(together[:, :-1], updated[:, :-1]))
        self.assertFalse(torch.equal(together[:, -1], updated[:, -1]))


if __name__ == "__main__":
    unittest.main()
