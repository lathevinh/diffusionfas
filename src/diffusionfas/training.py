"""Source-only training with bounded execution, EMA and safe checkpoint replay."""

from __future__ import annotations

from collections import defaultdict
import copy
import math
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor

from .data import DOMAINS, Video, addressed_seed
from .dataset import ClipCache, balanced_draws
from .io import object_digest, scientific_config
from .models import ClipDenoiser
from .scoring import residual_profile


class Budget:
    def __init__(self, specification: dict[str, Any], output: Path) -> None:
        for key in ("max_seconds", "max_optimizer_steps", "max_output_bytes"):
            value = specification[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"explicit positive execution budget required: {key}")
        self.specification, self.output = specification, output
        self.deadline = time.monotonic() + specification["max_seconds"]

    def check(self, steps: int, *, next_step: bool = False) -> None:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("training wall-clock budget exhausted; not a completed run")
        if steps + int(next_step) > self.specification["max_optimizer_steps"]:
            raise RuntimeError("optimizer-step budget exhausted; not a completed run")
        total = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())
        if total > self.specification["max_output_bytes"]:
            raise OSError("training output budget exhausted")


class EMA:
    def __init__(self, model: ClipDenoiser, decay: float) -> None:
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be inside (0,1)")
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: ClipDenoiser) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1 - self.decay)

    def install(self, model: ClipDenoiser) -> dict[str, Tensor]:
        previous = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    previous[name] = parameter.detach().clone()
                    parameter.copy_(self.shadow[name])
        return previous


def install_trainable(model: ClipDenoiser, state: dict[str, Tensor]) -> None:
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("checkpoint trainable parameter keys mismatch")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                if parameter.shape != state[name].shape or not torch.isfinite(state[name]).all():
                    raise ValueError("invalid checkpoint parameter shape/value")
                parameter.copy_(state[name])


def checkpoint_binding(config: dict[str, Any], videos: list[Video]) -> str:
    return object_digest({
        "config": scientific_config(config),
        "population": [v.to_dict() for v in sorted(videos, key=lambda v: v.key)],
    })


def save_checkpoint(path: Path, payload: dict[str, Any], budget: Budget, steps: int) -> None:
    budget.check(steps)
    temporary = path.with_suffix(".temporary")
    if temporary.exists():
        raise ValueError("unfinished checkpoint exists; explicit recovery required")
    limit = budget.specification["max_output_bytes"]
    current = sum(p.stat().st_size for p in budget.output.rglob("*") if p.is_file())

    class LimitedWriter:
        def __init__(self, stream: Any) -> None:
            self.stream = stream
            self.written = 0

        def write(self, data: bytes) -> int:
            if current + self.written + len(data) > limit:
                raise OSError("checkpoint would exceed output budget")
            count = self.stream.write(data)
            self.written += count
            return count

        def flush(self) -> None:
            self.stream.flush()

    with temporary.open("xb") as stream:
        torch.save(payload, LimitedWriter(stream))
        stream.flush()
        os.fsync(stream.fileno())
    budget.check(steps)
    os.replace(temporary, path)


def check_source_pools(videos: list[Video], target: str) -> tuple[list[Video], list[Video], dict[str, int]]:
    if target not in DOMAINS:
        raise ValueError("known outer target required")
    fitting, validation = [], []
    skipped: dict[str, int] = defaultdict(int)
    for video in videos:
        if video.dataset == target or video.partition not in {"fitting", "validation"} or video.label != "bona_fide":
            continue
        if video.status != "prepared":
            skipped[f"{video.dataset}/{video.partition}/{video.status}"] += 1
            continue
        (fitting if video.partition == "fitting" else validation).append(video)
    for source in set(DOMAINS) - {target}:
        for name, pool in (("fitting", fitting), ("validation", validation)):
            subset = [v for v in pool if v.dataset == source]
            if len(subset) < 2 or len({v.group for v in subset}) < 2:
                raise ValueError(f"insufficient prepared bonafide source support: {source}/{name}")
    return fitting, validation, dict(skipped)


def validation_metrics(
    model: ClipDenoiser, cache: ClipCache, videos: list[Video], device: torch.device, seed: int,
) -> dict[str, Any]:
    losses: dict[str, list[float]] = defaultdict(list)
    records = []
    for video in sorted(videos, key=lambda v: v.key):
        clip = cache.get(video)[None].to(device)
        levels = [0.5] if model.fixed_timestep is not None else [0.9, 0.5, 0.1]
        profile = residual_profile(model, clip, [video.key], alpha_bars=levels,
                                   draws=2, seed=seed)
        losses[video.dataset].append(float(profile.mean()))
        records.append({"key": video.key, "dataset": video.dataset,
                        "profile": profile.cpu().tolist()[0], "loss": float(profile.mean())})
    value = sum(sum(values) / len(values) for values in losses.values()) / len(losses)
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite source validation loss")
    return {"macro_loss": value,
            "by_domain": {name: {"originals": len(values), "mean_loss": sum(values) / len(values)}
                          for name, values in sorted(losses.items())},
            "records": records}


def validation_loss(
    model: ClipDenoiser, cache: ClipCache, videos: list[Video], device: torch.device, seed: int,
) -> float:
    return validation_metrics(model, cache, videos, device, seed)["macro_loss"]


def sampling_commitment(draws: list[Video], seed: int, epoch: int) -> str:
    return object_digest([[video.key, addressed_seed(seed, epoch, index, video.key)]
                          for index, video in enumerate(draws)])


def train(
    model: ClipDenoiser, videos: list[Video], config: dict[str, Any], output: Path,
    *, device: str = "cpu", resume: Path | None = None, stop_after_epochs: int | None = None,
) -> dict[str, Any]:
    if config["execution"]["source_training"] is not True:
        raise PermissionError("source training is not authorized in this configuration")
    if config["training"]["numeric"] != "float32":
        raise ValueError("only explicit FP32 training is implemented; no silent AMP fallback")
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
        raise ValueError("positive explicit pause epoch required")
    fitting, validation, skipped = check_source_pools(videos, config["target"])
    settings = config["training"]
    for key in ("effective_batch", "microbatch", "max_epochs", "minimum_epochs", "patience"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"positive integer training setting required: {key}")
    if settings["minimum_epochs"] > settings["max_epochs"]:
        raise ValueError("minimum_epochs exceeds max_epochs")
    if settings["learning_rate"] <= 0 or settings["weight_decay"] < 0 or settings["gradient_clip"] <= 0:
        raise ValueError("invalid optimizer settings")
    if output.exists() and resume is None:
        raise ValueError("fresh training requires a new output directory")
    output.mkdir(parents=True, exist_ok=resume is not None)
    budget = Budget(config["budget"], output)
    torch.manual_seed(config["seed"])
    target_device = torch.device(device)
    model.to(target_device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("frozen spatial baseline has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    ema = EMA(model, settings["ema_decay"])
    binding = checkpoint_binding(config, fitting + validation)
    epoch_start, steps, best_epoch, best_loss, history = 0, 0, -1, math.inf, []
    if resume is not None:
        state = torch.load(resume, map_location=target_device, weights_only=True)
        if state["binding"] != binding or state["complete"]:
            raise ValueError("resume requires matching incomplete source checkpoint")
        install_trainable(model, state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
        ema.shadow = state["ema"]
        epoch_start, steps = state["next_epoch"], state["steps"]
        best_epoch, best_loss, history = state["best_epoch"], state["best_loss"], state["history"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if target_device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    cache = ClipCache(model.resolution, config["clip"]["duration"], config["clip"]["fps"])
    for epoch in range(epoch_start, settings["max_epochs"]):
        budget.check(steps)
        warmup = settings["warmup_epochs"]
        if warmup < 0:
            raise ValueError("warmup_epochs cannot be negative")
        factor = (epoch + 1) / warmup if epoch < warmup else 0.5 * (
            1 + math.cos(math.pi * (epoch - warmup) / max(1, settings["max_epochs"] - warmup))
        )
        for group in optimizer.param_groups:
            group["lr"] = settings["learning_rate"] * factor
        draws = balanced_draws(fitting, config["seed"], epoch)
        model.train()
        loss_sum = 0.0
        for offset in range(0, len(draws), settings["effective_batch"]):
            batch = draws[offset:offset + settings["effective_batch"]]
            budget.check(steps, next_step=True)
            optimizer.zero_grad(set_to_none=True)
            for micro in range(0, len(batch), settings["microbatch"]):
                subset = batch[micro:micro + settings["microbatch"]]
                tensors = []
                noises, timesteps = [], []
                for index, video in enumerate(subset):
                    seed = addressed_seed(config["seed"], epoch, offset + micro + index, video.key)
                    clip = cache.get(video, seed, augment=True)
                    tensors.append(clip)
                    generator = torch.Generator(device=target_device).manual_seed(seed)
                    noises.append(torch.randn(clip.shape, device=target_device, generator=generator))
                    timesteps.append(
                        torch.randint(model.scheduler.config.num_train_timesteps,
                                      (1,), device=target_device, generator=generator)
                        if model.fixed_timestep is None else
                        torch.tensor([model.fixed_timestep], device=target_device, dtype=torch.long)
                    )
                clean = torch.stack(tensors).to(target_device)
                noise, t = torch.stack(noises), torch.cat(timesteps)
                noisy = model.scheduler.add_noise(clean, noise, t)
                prediction = model(noisy, t)
                loss = (prediction.float() - noise.float()).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                (loss * len(subset) / len(batch)).backward()
                loss_sum += float(loss.detach()) * len(subset)
                budget.check(steps)
            torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            ema.update(model)
        previous = ema.install(model)
        try:
            metrics = validation_metrics(model, cache, validation, target_device, config["seed"])
            val_loss = metrics["macro_loss"]
        finally:
            install_trainable(model, previous)
        budget.check(steps)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(draws), "validation_loss": val_loss,
                        "validation_by_domain": metrics["by_domain"],
                        "sampling_sha256": sampling_commitment(draws, config["seed"], epoch)})
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch
        complete = epoch + 1 == settings["max_epochs"] or (
            epoch + 1 >= settings["minimum_epochs"] and epoch - best_epoch >= settings["patience"]
        )
        payload = {
            "version": 1, "binding": binding, "complete": complete, "config": config,
            "parameters": {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad},
            "optimizer": optimizer.state_dict(), "ema": copy.deepcopy(ema.shadow),
            "next_epoch": epoch + 1, "steps": steps, "best_epoch": best_epoch,
            "best_loss": best_loss, "history": history, "skipped": skipped,
            "validation": metrics,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if target_device.type == "cuda" else [],
        }
        save_checkpoint(output / "last.pt", payload, budget, steps)
        if improved:
            save_checkpoint(output / "best.pt", payload, budget, steps)
        if complete:
            return {"complete": True, "epochs": epoch + 1, "steps": steps,
                    "best_epoch": best_epoch, "best_loss": best_loss, "skipped": skipped}
        if stop_after_epochs is not None and epoch + 1 >= stop_after_epochs:
            return {"complete": False, "paused_for_resume": True, "epochs": epoch + 1, "steps": steps}
    raise ValueError("resume has no remaining epochs; no successful run claimed")
