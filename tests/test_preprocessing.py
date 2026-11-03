from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from diffusionfas.data import Video
from diffusionfas.io import digest
from diffusionfas.preprocessing import MediaFailure, prepare_video


class FixtureFaces:
    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.array([4., 4., 28., 28.]), np.array([[10., 10.], [22., 10.],
                                                      [16., 16.], [11., 23.], [21., 23.]])

    def crop(self, rgb: np.ndarray, box: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        return cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_CUBIC)


class PreprocessingTests(unittest.TestCase):
    def test_real_synthetic_video_decode_hash_tracking_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "synthetic.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 16., (32, 32))
            self.assertTrue(writer.isOpened())
            for index in range(48):
                frame = np.full((32, 32, 3), index * 3, np.uint8)
                writer.write(frame)
            writer.release()
            capture = cv2.VideoCapture(str(path))
            hashes = []
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                hashes.append(hashlib.sha256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()).hexdigest())
            capture.release()
            row = Video(
                key="synthetic", dataset="CASIA-FASD", group="synthetic", subject="synthetic",
                partition="fitting", label="bona_fide", timestamps=tuple(i / 16 for i in range(48)),
                frame_hashes=tuple(hashes), media_sha256=digest(path), decoded_sha256=None,
                status="eligible_pending_tracking", media_path=str(path),
            )
            result = prepare_video(row, FixtureFaces(), root / "cache.npz")
            self.assertEqual(result.status, "prepared")
            with np.load(result.cache_path, allow_pickle=False) as cache:
                self.assertEqual(cache["frames"].shape, (24, 256, 256, 3))
                self.assertEqual(cache["native_orders"].shape, (24,))
            with self.assertRaises(MediaFailure):
                prepare_video(row, FixtureFaces(), root / "too-big.npz", max_output_bytes=100)
            fixed = prepare_video(row, FixtureFaces(), root / "fixed.npz",
                                  window_start=0.5, window_duration=2)
            with np.load(fixed.cache_path, allow_pickle=False) as cache:
                self.assertEqual(cache["frames"].shape, (16, 256, 256, 3))
                np.testing.assert_array_equal(cache["timestamps"], 0.5 + (np.arange(16) + 0.5) / 8)
            with self.assertRaises(ValueError):
                prepare_video(row, FixtureFaces(), root / "outside.npz", window_start=2, window_duration=2)
            with self.assertRaises(ValueError):
                prepare_video(row, FixtureFaces(), root / "half-request.npz", window_start=0.5)

            def native(path, consume, timeout):
                capture = cv2.VideoCapture(str(path))
                try:
                    index = 0
                    while True:
                        ok, bgr = capture.read()
                        if not ok:
                            break
                        consume(index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                        index += 1
                finally:
                    capture.release()

            replayed = prepare_video(row, FixtureFaces(), root / "replayed.npz",
                                     window_start=0.5, window_duration=2, native_replay=native)
            with np.load(fixed.cache_path, allow_pickle=False) as expected, np.load(replayed.cache_path, allow_pickle=False) as actual:
                np.testing.assert_array_equal(expected["frames"], actual["frames"])
            corrupted = replace(row, frame_hashes=("0" * 64, *row.frame_hashes[1:]))
            with self.assertRaisesRegex(MediaFailure, "native_RGB_hash"):
                prepare_video(corrupted, FixtureFaces(), root / "bad-native.npz",
                              window_start=0.5, window_duration=2, native_replay=native)


if __name__ == "__main__":
    unittest.main()
