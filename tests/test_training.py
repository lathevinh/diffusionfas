from __future__ import annotations

import copy
import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from diffusionfas.data import DOMAINS, Video, validate_isolation
from diffusionfas.dataset import ClipCache, balanced_draws
from diffusionfas.io import digest, read_json, write_json
from diffusionfas.cli import manifest_payload, run_profiles
from diffusionfas.models import scratch_model
from diffusionfas.training import check_source_pools, install_trainable, train

from test_models import TINY

ROOT = Path(__file__).resolve().parents[1]


def synthetic_population(root: Path) -> list[Video]:
    videos = []
    for domain in DOMAINS[1:]:
        for partition in ("fitting", "validation"):
            for index in range(2):
                number = len(videos) + 1
                path = root / f"cache-{number}.npz"
                frames = np.full((16, 256, 256, 3), number * 13, dtype=np.uint8)
                np.savez(path, frames=frames, timestamps=np.arange(16) / 8 + 0.0625)
                videos.append(Video(
                    key=f"{domain}/{partition}/{index}", dataset=domain,
                    group=f"{partition}/{index}", subject=f"{partition}/{index}",
                    partition=partition, label="bona_fide",
                    timestamps=tuple(np.arange(16) / 8 + 0.0625), frame_hashes=(),
                    media_sha256=None, decoded_sha256=None, status="prepared",
                    cache_path=str(path), cache_sha256=digest(path),
                ))
    return videos


def synthetic_config() -> dict:
    config = read_json(ROOT / "configs/offline-v1.json")
    config["seed"] = 42
    config["execution"]["source_training"] = True
    config["model"]["variant"] = "scratch_temporal"
    config["model"]["adapter_channels"] = 8
    config["scratch"] = TINY
    config["training"].update(effective_batch=3, microbatch=2, max_epochs=2,
                              minimum_epochs=1, patience=5, warmup_epochs=0)
    config["budget"] = {"max_seconds": 120, "max_optimizer_steps": 100,
                        "max_output_bytes": 64 * 1024**2}
    return config


class TrainingTests(unittest.TestCase):
    def test_synthetic_train_reload_and_exact_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = synthetic_population(root)
            validate_isolation(videos)
            config = synthetic_config()
            torch.manual_seed(7)
            template = scratch_model(TINY, temporal=True, adapter_channels=8)
            full = copy.deepcopy(template)
            report = train(full, videos, config, root / "full")
            self.assertTrue(report["complete"])
            self.assertEqual(report["steps"], 4)
            partial_config = copy.deepcopy(config)
            partial_config["budget"]["max_optimizer_steps"] = 2
            partial = copy.deepcopy(template)
            with self.assertRaisesRegex(RuntimeError, "optimizer-step budget"):
                train(partial, videos, partial_config, root / "partial")
            state = torch.load(root / "partial/last.pt", weights_only=True)
            self.assertFalse(state["complete"])
            resumed = copy.deepcopy(template)
            result = train(resumed, videos, config, root / "partial", resume=root / "partial/last.pt")
            self.assertTrue(result["complete"])
            full_state = torch.load(root / "full/last.pt", weights_only=True)
            resumed_state = torch.load(root / "partial/last.pt", weights_only=True)
            self.assertEqual(full_state["history"], resumed_state["history"])
            self.assertTrue(all(torch.equal(tensor, resumed_state["parameters"][name])
                                for name, tensor in full_state["parameters"].items()))
            reloaded = copy.deepcopy(template)
            install_trainable(reloaded, full_state["ema"])
            comparison = copy.deepcopy(template)
            install_trainable(comparison, resumed_state["ema"])
            clip = ClipCache(16).get(videos[0])[None]
            self.assertTrue(torch.equal(reloaded(clip, torch.tensor([2])),
                                        comparison(clip, torch.tensor([2]))))
            config["execution"]["source_scoring"] = True
            write_json(root / "config.json", config)
            write_json(root / "manifest.json", manifest_payload(videos, {"synthetic": True}))
            run_profiles(argparse.Namespace(
                config=root / "config.json", manifest=root / "manifest.json", partition="validation",
                include_attacks=False, middle_frame=False, checkpoint_root=root / "full",
                device="cpu", out=root / "profiles.json",
            ))
            profiles = read_json(root / "profiles.json")
            self.assertEqual(len(profiles["records"]), 6)
            self.assertTrue(all(r["profile"] is not None for r in profiles["records"]))

    def test_no_training_without_approval_or_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = synthetic_population(root)
            config = synthetic_config()
            model = scratch_model(TINY, temporal=False)
            config["execution"]["source_training"] = False
            with self.assertRaises(PermissionError):
                train(model, videos, config, root / "denied")
            self.assertFalse((root / "denied").exists())
            config["execution"]["source_training"] = True
            config["budget"]["max_seconds"] = None
            with self.assertRaises(ValueError):
                train(model, videos, config, root / "no-budget")

    def test_source_pools_cannot_use_target_or_attacks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = synthetic_population(Path(directory))
            fitting, validation, _ = check_source_pools(rows, "OULU-NPU")
            self.assertEqual(len(fitting), 6)
            self.assertEqual(len(validation), 6)
            bad = [replace(row, label="attack") for row in rows]
            with self.assertRaises(ValueError):
                check_source_pools(bad, "OULU-NPU")

    def test_cache_integrity_and_consistent_augmentation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = synthetic_population(Path(directory))
            cache = ClipCache(16)
            self.assertTrue(torch.equal(cache.get(rows[0], 4, True), cache.get(rows[0], 4, True)))
            with self.assertRaises(ValueError):
                cache.get(replace(rows[0], cache_sha256="changed"))
            draws = balanced_draws(rows[:2] + rows[4:6] + rows[8:10], 42, 0)
            counts = {domain: sum(v.dataset == domain for v in draws) for domain in DOMAINS[1:]}
            self.assertEqual(set(counts.values()), {2})

    def test_native_timestamp_jitter_does_not_create_random_window_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "jitter.npz"
            grid = np.arange(24) / 8 + 0.0625
            native = grid + np.where(np.arange(24) % 2, 0.015, -0.015)
            np.savez(path, frames=np.zeros((24, 256, 256, 3), dtype=np.uint8),
                     timestamps=grid, native_timestamps=native)
            row = Video(
                key="jitter", dataset="CASIA-FASD", group="jitter", subject="jitter",
                partition="fitting", label="bona_fide", timestamps=tuple(native), frame_hashes=(),
                media_sha256=None, decoded_sha256=None, status="prepared",
                cache_path=str(path), cache_sha256=digest(path),
            )
            cache = ClipCache(16)
            for seed in range(20):
                self.assertEqual(cache.get(row, seed).shape, (16, 3, 16, 16))

    def test_odd_cache_grid_with_nonbinary_origin_has_distinct_contiguous_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odd-grid.npz"
            grid = tuple(0.1 + (index + 0.5) / 8 for index in range(17))
            frames = np.broadcast_to(np.arange(17, dtype=np.uint8)[:, None, None, None],
                                     (17, 256, 256, 3)).copy()
            np.savez(path, frames=frames, timestamps=grid)
            row = Video(
                key="odd-grid", dataset="CASIA-FASD", group="odd-grid", subject="odd-grid",
                partition="fitting", label="bona_fide", timestamps=grid, frame_hashes=(),
                media_sha256=None, decoded_sha256=None, status="prepared",
                cache_path=str(path), cache_sha256=digest(path),
            )
            cache = ClipCache(16)

            def frame_numbers(seed=None):
                clip = cache.get(row, seed)
                self.assertEqual(clip.shape, (16, 3, 16, 16))
                return ((clip[:, 0, 0, 0] + 1) * 255 / 2).round().int()

            self.assertTrue(torch.equal(frame_numbers(), torch.arange(16)))
            starts = set()
            for seed in range(20):
                numbers = frame_numbers(seed)
                starts.add(int(numbers[0]))
                self.assertTrue(torch.equal(numbers, torch.arange(int(numbers[0]), int(numbers[0]) + 16)))
                self.assertTrue(torch.equal(numbers, frame_numbers(seed)))
            self.assertEqual(starts, {0, 1})
            with self.assertRaisesRegex(ValueError, "cached clip shorter"):
                ClipCache(16, duration=3).get(row)

    def test_cache_sampling_requires_a_positive_integral_frame_count(self) -> None:
        for duration, fps in ((0, 8), (-1, 8), (float("nan"), 8), (0.3, 8), (2, 0)):
            with self.subTest(duration=duration, fps=fps), self.assertRaises(ValueError):
                ClipCache(16, duration=duration, fps=fps)


if __name__ == "__main__":
    unittest.main()
