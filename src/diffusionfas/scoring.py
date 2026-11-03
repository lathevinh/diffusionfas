"""Deterministic multi-noise residuals and source-only robust normalization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .data import addressed_seed
from .models import ClipDenoiser


@torch.no_grad()
def residual_profile(
    model: ClipDenoiser, clips: Tensor, keys: list[str], *,
    alpha_bars: list[float], draws: int, seed: int, frame_addresses: list[int] | None = None,
) -> Tensor:
    if clips.ndim != 5 or len(keys) != clips.shape[0] or draws < 1:
        raise ValueError("one key per clip and positive noise draws required")
    if not alpha_bars or any(not 0 < a < 1 for a in alpha_bars):
        raise ValueError("noise alpha_bars must be inside (0,1)")
    if not torch.isfinite(clips).all():
        raise ValueError("finite scoring clips required")
    addresses = list(range(clips.shape[1])) if frame_addresses is None else frame_addresses
    if len(addresses) != clips.shape[1] or len(set(addresses)) != len(addresses):
        raise ValueError("unique frame address per observed frame required")
    was_training = model.training
    model.eval()
    result = []
    try:
        for clip, key in zip(clips, keys, strict=True):
            levels = []
            for level, alpha in enumerate(alpha_bars):
                timestep = int((model.scheduler.alphas_cumprod - alpha).abs().argmin())
                t = torch.tensor([timestep], device=clips.device, dtype=torch.long)
                errors = []
                for draw in range(draws):
                    frame_noises = []
                    for address in addresses:
                        generator = torch.Generator(device=clips.device).manual_seed(
                            addressed_seed(seed, key, level, draw, address)
                        )
                        frame_noises.append(torch.randn(
                            clip.shape[1:], device=clip.device, dtype=clip.dtype, generator=generator,
                        ))
                    noise = torch.stack(frame_noises)[None]
                    noisy = model.scheduler.add_noise(clip[None], noise, t)
                    predicted = model(noisy, t)
                    errors.append((predicted.float() - noise.float()).square().mean())
                levels.append(torch.stack(errors).mean())
            result.append(torch.stack(levels))
    finally:
        model.train(was_training)
    profile = torch.stack(result)
    if not torch.isfinite(profile).all():
        raise FloatingPointError("nonfinite denoising residual profile")
    return profile


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.ndim != 1 or weights.shape != values.shape or values.size == 0:
        raise ValueError("nonempty aligned scalar values and weights required")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("finite values and positive weights required")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be inside [0,1]")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left")), len(order) - 1)
    return float(values[order[index]])


@dataclass(frozen=True)
class RobustNormalizer:
    median: tuple[float, ...]
    iqr: tuple[float, ...]
    sources: tuple[str, ...]

    @classmethod
    def fit(cls, profiles: np.ndarray, records: list[dict[str, Any]], target: str) -> RobustNormalizer:
        profiles = np.asarray(profiles, dtype=np.float64)
        if profiles.ndim != 2 or len(records) != len(profiles) or not np.isfinite(profiles).all():
            raise ValueError("finite aligned residual profiles required")
        if not records or any(
            r["dataset"] == target or r["partition"] != "validation" or r["label"] != "bona_fide"
            for r in records
        ):
            raise ValueError("normalization requires only source validation bonafide")
        sources = tuple(sorted({r["dataset"] for r in records}))
        counts = {s: sum(r["dataset"] == s for r in records) for s in sources}
        weights = np.array([1 / counts[r["dataset"]] for r in records], dtype=np.float64)
        medians, scales = [], []
        for column in profiles.T:
            median = weighted_quantile(column, weights, 0.5)
            scale = weighted_quantile(column, weights, 0.75) - weighted_quantile(column, weights, 0.25)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("zero/nonfinite validation IQR; normalization blocked")
            medians.append(median)
            scales.append(scale)
        return cls(tuple(medians), tuple(scales), sources)

    def transform(self, profiles: np.ndarray) -> np.ndarray:
        profiles = np.asarray(profiles, dtype=np.float64)
        if (
            profiles.ndim != 2 or profiles.shape[1] != len(self.median)
            or not np.isfinite(profiles).all() or any(s <= 0 or not math.isfinite(s) for s in self.iqr)
        ):
            raise ValueError("finite matching profiles and positive normalization scales required")
        return ((profiles - np.array(self.median)) / np.array(self.iqr)).mean(axis=1)
