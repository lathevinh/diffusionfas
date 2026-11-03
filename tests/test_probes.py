from __future__ import annotations

from contextlib import contextmanager
import io
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from diffusionfas.acquisition import convert_state, download
from diffusionfas.models import ClipDenoiser, build_model, scratch_model
from diffusionfas.source_probe import copy_member, safe_member
from diffusionfas.supervision import run_bounded
from test_models import TINY


class ProbeTests(unittest.TestCase):
    def test_lossless_legacy_attention_conversion(self) -> None:
        old = {"down_blocks.0.attentions.0.query.weight": torch.arange(16.).reshape(4, 4, 1, 1),
               "other.bias": torch.zeros(4)}
        expected = {"down_blocks.0.attentions.0.to_q.weight": torch.empty(4, 4),
                    "other.bias": torch.empty(4)}
        result, receipt = convert_state(old, expected)
        self.assertTrue(torch.equal(result["down_blocks.0.attentions.0.to_q.weight"], torch.arange(16.).reshape(4, 4)))
        self.assertEqual(receipt, {"renamed_attention_tensors": 1, "reshaped_1x1_attention_tensors": 1})
        for invalid in (
            {"other.bias": torch.full((4,), float("nan"))},
            {**old, "unknown": torch.zeros(1)},
            {**old, "down_blocks.0.attentions.0.to_q.weight": torch.zeros(4, 4)},
            {"other.bias": torch.zeros(3)},
            {"other.bias": torch.zeros(4)},
        ):
            with self.assertRaises(ValueError):
                convert_state(invalid, expected)

    def test_download_byte_cap_and_hash_without_network(self) -> None:
        @contextmanager
        def response(*args, **kwargs):
            stream = io.BytesIO(b"synthetic-checkpoint")
            stream.url = "https://huggingface.co/synthetic"
            yield stream

        with tempfile.TemporaryDirectory() as directory, patch(
            "diffusionfas.acquisition.urllib.request.urlopen", response,
        ):
            root = Path(directory)
            with self.assertRaises(ValueError):
                download("https://huggingface.co/synthetic", root / "limited",
                         maximum_bytes=3, deadline=time.monotonic() + 5)
            with self.assertRaises(ValueError):
                download("https://huggingface.co/synthetic", root / "bad-hash",
                         maximum_bytes=100, deadline=time.monotonic() + 5, expected_sha256="bad")

    def test_bounded_subprocess_success_timeout_failure_and_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def execute(code: str, name: str, **limits):
                return run_bounded([sys.executable, "-c", code], cwd=root, owned_root=root,
                                   log=root / name, max_rss_bytes=512 * 1024**2,
                                   max_disk_bytes=limits.get("disk", 1024**2),
                                   max_seconds=limits.get("seconds", 5))

            self.assertTrue(execute("print('ok')", "success.log")["completed"])
            self.assertFalse(execute("raise ValueError('explicit')", "failure.log")["completed"])
            timeout = execute("import time; time.sleep(20)", "timeout.log", seconds=0.25)
            self.assertEqual(timeout["guard_failure"], "wall_clock_cap")
            self.assertLess(timeout["wall_seconds"], 6)
            disk = execute("from pathlib import Path; import time; Path('large').write_bytes(b'x'*10000); time.sleep(20)",
                           "disk.log", disk=1000)
            self.assertEqual(disk["guard_failure"], "owned_disk_cap")

    def test_bounded_subprocess_kills_owned_descendants_after_leader_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = (
                "import subprocess,sys,time; from pathlib import Path; "
                "child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(20)']); "
                "Path('child-pid').write_text(str(child.pid)); time.sleep(0.3)"
            )
            result = run_bounded([sys.executable, "-c", code], cwd=root, owned_root=root,
                                 log=root / "parent.log", max_seconds=5,
                                 max_rss_bytes=512 * 1024**2, max_disk_bytes=1024**2)
            self.assertTrue(result["completed"])
            child = int((root / "child-pid").read_text())
            for _ in range(20):
                status = Path(f"/proc/{child}/stat")
                if not status.exists() or status.read_text().split()[2] == "Z":
                    break
                time.sleep(0.05)
            else:
                self.fail("owned descendant remained alive after supervisor cleanup")

    def test_bounded_subprocess_counts_predecessor_output_against_combined_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous, current = root / "previous", root / "current"
            previous.mkdir()
            current.mkdir()
            (previous / "retained-cache").write_bytes(b"x" * 2048)
            result = run_bounded(
                [sys.executable, "-c", "import time; time.sleep(20)"], cwd=current,
                owned_root=current, log=current / "batch.log", max_seconds=5,
                max_rss_bytes=512 * 1024**2, max_disk_bytes=1024,
                additional_output_roots=(previous,),
            )
            self.assertFalse(result["completed"])
            self.assertEqual(result["guard_failure"], "owned_disk_cap")
            self.assertGreaterEqual(result["sampled_owned_peak_disk_bytes"], 2048)

    def test_archive_members_are_exact_and_media_hash_checked(self) -> None:
        for member in ("../escape", "/absolute", "foo/*", "foo/?.avi", "-option", "foo\\bar"):
            with self.assertRaises(ValueError):
                safe_member(member)
        safe_member("release/videos/clip.mov")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                copy_member(io.BytesIO(b"bad"), Path(directory) / "media", 3, "incorrect", time.monotonic() + 2)

    def test_frozen_frame_microbatch_equivalence_gradients_and_configuration(self) -> None:
        torch.manual_seed(2)
        base = scratch_model(TINY, temporal=False)
        model = ClipDenoiser(base.backbone, base.scheduler, temporal=True,
                            freeze_backbone=True, adapter_channels=8)
        noisy, timestep = torch.randn(1, 5, 3, 16, 16), torch.tensor([2])
        expected = model(noisy, timestep)
        model.backbone_frame_batch = 2
        actual = model(noisy, timestep)
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=2e-6)
        actual.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.adapter.parameters()))
        model.backbone_frame_batch = 0
        with self.assertRaises(ValueError):
            model(noisy, timestep)
        configured = build_model({"model": {"variant": "scratch_spatial", "adapter_channels": 8,
                                             "backbone_frame_batch": 2}, "scratch": TINY})
        self.assertEqual(configured.backbone_frame_batch, 2)
