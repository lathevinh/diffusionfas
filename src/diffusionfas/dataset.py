"""Verified local cache access with addressed, video-balanced windows."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .data import Video, addressed_seed, window_indices
from .io import digest


class ClipCache:
    def __init__(self, resolution: int, duration: float = 2, fps: int = 8) -> None:
        self.resolution, self.duration, self.fps = resolution, duration, fps
        self.verified: dict[str, tuple[int, int, str]] = {}

    def get(self, video: Video, seed: int | None = None, augment: bool = False) -> Tensor:
        if video.status != "prepared" or not video.cache_path or not video.cache_sha256:
            raise ValueError("prepared, hash-bound cache required")
        path = Path(video.cache_path)
        stat = path.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns, video.cache_sha256)
        if self.verified.get(str(path)) != fingerprint:
            if digest(path) != video.cache_sha256:
                raise ValueError("clip cache digest mismatch")
            self.verified[str(path)] = fingerprint
        with np.load(path, allow_pickle=False) as cache:
            frames, timestamps = cache["frames"], tuple(float(t) for t in cache["timestamps"])
            if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (256, 256, 3):
                raise ValueError("cache frames must be uint8 RGB256")
            if len(frames) != len(timestamps) or len(timestamps) < 2:
                raise ValueError("cache timestamp/frame mismatch")
            if "native_timestamps" in cache:
                native = cache["native_timestamps"]
                if native.shape != (len(timestamps),) or not np.isfinite(native).all():
                    raise ValueError("finite native timestamp per cached frame required")
                if np.any(np.abs(native - np.array(timestamps)) > 0.5 / self.fps + 1e-8):
                    raise ValueError("native frame outside bound sampling tolerance")
            gaps = np.diff(timestamps)
            if not np.isfinite(gaps).all() or np.any(gaps <= 0):
                raise ValueError("cache timestamps must increase")
            if not np.allclose(gaps, 1 / self.fps, rtol=0, atol=1e-7):
                raise ValueError("cache must retain its uniform requested sampling grid")
            low, high = timestamps[0] - 0.5 / self.fps, timestamps[-1] + 0.5 / self.fps - self.duration
            if high + 1e-8 < low:
                raise ValueError("cached clip shorter than requested duration")
            start = (low + high) / 2 if seed is None else random.Random(seed).uniform(low, max(low, high))
            indexes = window_indices(timestamps, start, self.duration, self.fps)
            tensor = torch.from_numpy(frames[list(indexes)].copy()).permute(0, 3, 1, 2).float() / 255
        tensor = F.interpolate(tensor, (self.resolution, self.resolution), mode="bilinear",
                               align_corners=False, antialias=True)
        if augment:
            if seed is None:
                raise ValueError("addressed augmentation seed required")
            rng = random.Random(addressed_seed(seed, "augmentation"))
            if rng.random() < 0.5:
                tensor = tensor.flip(-1)
            exposure, gamma = 2 ** rng.uniform(-0.3, 0.3), rng.uniform(0.9, 1.1)
            gains = tensor.new_tensor([rng.uniform(0.95, 1.05) for _ in range(3)])[None, :, None, None]
            tensor = (tensor * exposure * gains).clamp(0, 1).pow(gamma)
        return tensor.mul(2).sub(1)


def balanced_draws(videos: list[Video], seed: int, epoch: int) -> list[Video]:
    if not videos:
        raise ValueError("nonempty source bonafide fitting pool required")
    sources = sorted({v.dataset for v in videos})
    pools = {source: sorted((v for v in videos if v.dataset == source), key=lambda v: v.key) for source in sources}
    rng = random.Random(addressed_seed(seed, "epoch", epoch))
    result = []
    while len(result) < len(videos):
        order = sources.copy()
        rng.shuffle(order)
        for source in order:
            if len(result) == len(videos):
                break
            result.append(rng.choice(pools[source]))
    return result
