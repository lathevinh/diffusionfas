"""Local-only denoisers and a zero-initialized temporal residual adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from diffusers import DDPMScheduler, UNet2DModel
from safetensors.torch import load_file

from .io import digest, read_json


class TemporalAdapter(nn.Module):
    def __init__(self, channels: int, train_timesteps: int, temporal_kernel: int = 3) -> None:
        super().__init__()
        if channels < 8 or channels % 8:
            raise ValueError("adapter channels must be a positive multiple of eight")
        if temporal_kernel not in {1, 3}:
            raise ValueError("adapter temporal kernel must be one or three")
        self.encoder = nn.Sequential(
            nn.Conv2d(6, channels, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.time = nn.Embedding(train_timesteps, channels)
        self.norm = nn.GroupNorm(8, channels)
        self.temporal = nn.Sequential(
            nn.SiLU(),
            nn.Conv3d(channels, channels, (temporal_kernel, 1, 1), padding=(temporal_kernel // 2, 0, 0)),
            nn.SiLU(), nn.Conv3d(channels, channels, 1), nn.SiLU(),
        )
        self.output = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy: Tensor, base_noise: Tensor, timestep: Tensor) -> Tensor:
        batch, frames, _, height, width = noisy.shape
        features = self.norm(self.encoder(torch.cat((noisy, base_noise), dim=2).flatten(0, 1)))
        _, channels, low_h, low_w = features.shape
        features = features.reshape(batch, frames, channels, low_h, low_w).transpose(1, 2)
        features = features + self.time(timestep)[:, :, None, None, None]
        features = self.temporal(features).transpose(1, 2).flatten(0, 1)
        residual = self.output(features)
        residual = F.interpolate(residual, (height, width), mode="bilinear", align_corners=False)
        return residual.reshape(batch, frames, 3, height, width)


class ClipDenoiser(nn.Module):
    def __init__(
        self, backbone: UNet2DModel, scheduler: DDPMScheduler,
        *, temporal: bool, freeze_backbone: bool, adapter_channels: int = 32,
        temporal_kernel: int = 3,
    ) -> None:
        super().__init__()
        if scheduler.config.prediction_type != "epsilon":
            raise ValueError("epsilon-prediction scheduler required")
        self.backbone = backbone
        self.scheduler = scheduler
        self.freeze_backbone = freeze_backbone
        self.fixed_timestep: int | None = None
        sample_size = backbone.config.sample_size
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("square native sample_size required")
        self.resolution = sample_size
        self.adapter = TemporalAdapter(
            adapter_channels, scheduler.config.num_train_timesteps, temporal_kernel,
        ) if temporal else None
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> ClipDenoiser:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, noisy: Tensor, timestep: Tensor) -> Tensor:
        if noisy.ndim != 5 or noisy.shape[2:] != (3, self.resolution, self.resolution):
            raise ValueError("input must be B,T,3,native_resolution,native_resolution")
        if timestep.shape != (noisy.shape[0],) or timestep.dtype != torch.long:
            raise ValueError("one integer timestep per clip required")
        if torch.any(timestep < 0) or torch.any(timestep >= self.scheduler.config.num_train_timesteps):
            raise ValueError("timestep outside scheduler range")
        if self.fixed_timestep is not None and not torch.all(timestep == self.fixed_timestep):
            raise ValueError("fixed-noise denoiser received a different noise level")
        flattened = noisy.flatten(0, 1)
        conditioning = torch.zeros_like(timestep) if self.fixed_timestep is not None else timestep
        repeated = conditioning.repeat_interleave(noisy.shape[1])
        if self.freeze_backbone:
            with torch.no_grad():
                base = self.backbone(flattened, repeated).sample
        else:
            base = self.backbone(flattened, repeated).sample
        base = base.reshape_as(noisy)
        return base if self.adapter is None else base + self.adapter(noisy, base, timestep)


def scratch_model(
    specification: dict[str, Any], *, temporal: bool, adapter_channels: int = 32,
) -> ClipDenoiser:
    channels = tuple(specification["channels"])
    layers = specification["layers_per_block"]
    if not channels or any(c < 8 or c % 8 for c in channels):
        raise ValueError("scratch channels must be positive multiples of eight")
    depth = len(channels)
    if specification["resolution"] % 2 ** (depth - 1):
        raise ValueError("scratch resolution must divide U-Net downsampling factor")
    backbone = UNet2DModel(
        sample_size=specification["resolution"], in_channels=3, out_channels=3,
        block_out_channels=channels, layers_per_block=layers, norm_num_groups=8,
        down_block_types=("DownBlock2D",) * depth,
        up_block_types=("UpBlock2D",) * depth,
        add_attention=False,
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=specification["train_timesteps"],
        beta_schedule=specification["beta_schedule"], prediction_type="epsilon",
    )
    return ClipDenoiser(backbone, scheduler, temporal=temporal, freeze_backbone=False,
                        adapter_channels=adapter_channels)


def local_pretrained_model(
    bundle: Path, manifest_sha256: str, *, temporal: bool, adapter_channels: int = 32,
    temporal_kernel: int = 3,
) -> ClipDenoiser:
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("regular local pretrained bundle directory required")
    manifest = read_json(bundle / "manifest.json", manifest_sha256)
    expected_names = {"config.json", "scheduler_config.json", "model.safetensors"}
    if set(manifest["files"]) != expected_names or not manifest.get("upstream_revision"):
        raise ValueError("bundle must bind config, scheduler and safe weights with upstream revision")
    if manifest.get("rights_reviewed") is not True:
        raise ValueError("local pretrained bundle rights review is required")
    for name in expected_names:
        if digest(bundle / name) != manifest["files"][name]:
            raise ValueError("pretrained bundle file hash mismatch")
    backbone = UNet2DModel.from_config(read_json(bundle / "config.json"))
    weights = load_file(str(bundle / "model.safetensors"), device="cpu")
    backbone.load_state_dict(weights, strict=True)
    scheduler = DDPMScheduler.from_config(read_json(bundle / "scheduler_config.json"))
    return ClipDenoiser(backbone, scheduler, temporal=temporal, freeze_backbone=True,
                        adapter_channels=adapter_channels, temporal_kernel=temporal_kernel)


def build_model(config: dict[str, Any]) -> ClipDenoiser:
    model = config["model"]
    variant = model["variant"]
    if variant == "scratch_fixed_noise":
        if config["scoring"]["alpha_bars"] != [0.5]:
            raise ValueError("fixed-noise baseline requires scoring.alpha_bars=[0.5]")
        result = scratch_model(config["scratch"], temporal=False)
        result.fixed_timestep = int((result.scheduler.alphas_cumprod - 0.5).abs().argmin())
        return result
    if variant in {"scratch_spatial", "scratch_temporal"}:
        return scratch_model(config["scratch"], temporal=variant.endswith("temporal"),
                             adapter_channels=model["adapter_channels"])
    if variant in {"pretrained_spatial", "pretrained_spatial_adapter", "pretrained_temporal"}:
        if not model["local_bundle"] or not model["bundle_manifest_sha256"]:
            raise ValueError("pretrained weights unavailable: provide a verified LOCAL safetensors bundle; no automatic download")
        return local_pretrained_model(
            Path(model["local_bundle"]), model["bundle_manifest_sha256"],
            temporal=variant != "pretrained_spatial", adapter_channels=model["adapter_channels"],
            temporal_kernel=1 if variant == "pretrained_spatial_adapter" else 3,
        )
    raise ValueError("unknown model variant")
