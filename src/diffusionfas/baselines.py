"""One-class Gaussian baselines for receipt-bound, locally computed frame features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf

from .data import Video
from .io import digest


class FrozenDino:
    def __init__(self, source: Path, revision: str, checkpoint: Path, checkpoint_sha256: str, device: str) -> None:
        import torch

        if source.is_symlink() or not source.is_dir() or digest(checkpoint) != checkpoint_sha256:
            raise ValueError("verified local DINO source and checkpoint required")
        observed = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        clean = subprocess.run(["git", "-C", str(source), "diff", "--no-ext-diff", "--quiet", "HEAD", "--"],
                               check=False, capture_output=True, timeout=10)
        if observed != revision or clean.returncode != 0:
            raise ValueError("DINO source revision or tracked worktree differs")
        sys.path.insert(0, str(source))
        from dinov2.hub import backbones

        self.encoder = backbones.dinov2_vitb14_reg(pretrained=False)
        self.encoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.encoder.to(device).eval().requires_grad_(False)
        self.device = device

    def encode(self, clip: Any) -> np.ndarray:
        import torch

        if clip.ndim != 4 or clip.shape[1:] != (3, 224, 224):
            raise ValueError("DINO baseline requires T,3,224,224 clip")
        rgb = clip.to(self.device).add(1).div(2)
        mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        features = []
        with torch.no_grad():
            for frame in ((rgb - mean) / std).split(1):
                result = self.encoder.forward_features(frame)
                features.append(torch.cat((result["x_norm_clstoken"],
                                           result["x_norm_patchtokens"].mean(1)), dim=1))
        return torch.cat(features).cpu().numpy()


def clip_feature(features: np.ndarray, *, temporal: bool) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) < 2 or not np.isfinite(features).all():
        raise ValueError("finite T,D features from at least two observed frames required")
    mean = features.mean(axis=0)
    if not temporal:
        return mean
    differences = np.abs(np.diff(features, axis=0))
    return np.concatenate((mean, differences.mean(axis=0), differences.std(axis=0)))


@dataclass(frozen=True)
class FeatureGaussian:
    location: np.ndarray
    precision: np.ndarray
    sources: tuple[str, ...]

    @classmethod
    def fit(cls, features: np.ndarray, videos: list[Video], target: str) -> FeatureGaussian:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or len(features) != len(videos) or not np.isfinite(features).all():
            raise ValueError("finite aligned video features required")
        if not videos or any(v.dataset == target or v.partition != "fitting" or v.label != "bona_fide" for v in videos):
            raise ValueError("feature baseline fits only source fitting bonafide")
        sources = tuple(sorted({v.dataset for v in videos}))
        # Equal source weight through source-normalized first and second moments.
        center = np.mean([features[[v.dataset == source for v in videos]].mean(0) for source in sources], axis=0)
        centered = features - center
        counts = {source: sum(v.dataset == source for v in videos) for source in sources}
        weighted = np.stack([
            row * np.sqrt(len(videos) / (len(sources) * counts[video.dataset]))
            for row, video in zip(centered, videos, strict=True)
        ])
        estimator = LedoitWolf(assume_centered=True).fit(weighted)
        if not np.isfinite(estimator.precision_).all():
            raise ValueError("nonfinite Gaussian precision")
        return cls(center, estimator.precision_, sources)

    def score(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1:] != self.location.shape or not np.isfinite(features).all():
            raise ValueError("finite matching feature matrix required")
        centered = features - self.location
        return np.einsum("ni,ij,nj->n", centered, self.precision, centered)


def read_feature_receipt(path: Path, expected_sha256: str, receipt: dict[str, Any]) -> np.ndarray:
    if digest(path) != expected_sha256 or not receipt.get("encoder_sha256") or not receipt.get("preprocessing_sha256"):
        raise ValueError("frame features require array, encoder and preprocessing commitments")
    if receipt.get("external_pretraining") is not True:
        raise ValueError("frozen pretrained feature baseline must disclose external pretraining")
    with np.load(path, allow_pickle=False) as data:
        features = data["features"]
        if features.ndim != 3 or not np.isfinite(features).all():
            raise ValueError("finite N,T,D frame features required")
        return features.copy()
