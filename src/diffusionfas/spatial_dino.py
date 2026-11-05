"""Soft anatomical patch pooling on the existing full-DINO PAD reference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import gc
import hashlib
import importlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.nn import functional as F

from .io import object_digest
from .mat_input import POLICY as MAT_POLICY
from .regional_dino import RegionSupportError, crop_landmarks

ARMS = ("reference", "uniform", "content", "anatomical", "permuted")
SEED = 20260917
SIZE, PATCH, TOKENS, DIMENSION = 224, 14, 256, 768
REGIONS = ("left_eye", "right_eye", "nose", "mouth")
SETTINGS = {
    "gate_hidden": 32, "gate_logit_bound": 2., "uniform_floor": .25,
    "prior_strength": 1., "prior": "one_plus_mean_four_eye_distance_gaussians",
    "control": "single_fixed_nonidentity_patch_permutation_after_augmentation_and_PDA",
    "fusion": "unchanged_CLS_head_plus_zero_initialized_linear_weighted_patch_head",
    "objective": "unchanged_global_CE_plus_original_detached_CLS_attention_APL_no_tail",
    "epochs": 200, "patience": 20, "batch": 32,
    "backbone_lr": 5e-6, "head_lr": 5e-5, "weight_decay": .05,
    "selection": "source_equal_domain_equal_class_BCE_earliest_strict_best_no_refit",
    "geometry": "historical_landmarks_shared_context_RGB224_pixel_center_coordinates",
}


@dataclass(frozen=True)
class SourceModules:
    model: ModuleType
    augmentation: ModuleType


def source_modules(root: Path) -> SourceModules:
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root/"scripts/paper329"), str(root/"scripts")]
    modules = {}
    for name in ("model", "augmentation"):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != (root/f"scripts/paper329/{name}.py").resolve():
            raise ValueError(f"unbound reference module: {name}")
        modules[name] = module
    for name, relative in (("assets", "scripts/paper329/assets.py"),
                           ("objective", "scripts/paper327/objective.py")):
        if Path(sys.modules[name].__file__).resolve() != (root/relative).resolve():
            raise ValueError(f"unbound reference dependency: {name}")
    return SourceModules(modules["model"], modules["augmentation"])


def points224(transaction: dict[str, Any]) -> np.ndarray:
    points = (crop_landmarks(transaction)+.5)*(SIZE/518)-.5
    if (np.any(points < -.5) or np.any(points > SIZE-.5)
            or np.linalg.norm(points[1]-points[0]) < 1):
        raise RegionSupportError("spatial_prior_invalid_landmark_support")
    return points


def prior_image(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if (points.shape != (5, 2) or not np.isfinite(points).all()
            or np.any(points < -.5) or np.any(points > SIZE-.5)):
        raise ValueError("five in-crop finite RGB224 landmarks required")
    eye = points[1]-points[0]
    distance = float(np.linalg.norm(eye))
    if distance < 1:
        raise RegionSupportError("spatial_prior_degenerate_eyes")
    horizontal, vertical = eye/distance, np.array([-eye[1], eye[0]])/distance
    y, x = np.indices((SIZE, SIZE))
    centers = (points[0], points[1], points[2], points[3:5].mean(axis=0))
    result = np.ones((SIZE, SIZE), dtype=np.float64)
    for region, center in zip(REGIONS, centers, strict=True):
        rx, ry = np.asarray(MAT_POLICY["ellipse_radii_in_eye_distances"][region])*distance
        dx, dy = x-center[0], y-center[1]
        radius = ((dx*horizontal[0]+dy*horizontal[1])/rx)**2
        radius += ((dx*vertical[0]+dy*vertical[1])/ry)**2
        result += np.exp(-.5*radius)/len(REGIONS)
    return result.astype(np.float32)


def patch_prior(images: np.ndarray) -> np.ndarray:
    if (images.ndim != 3 or images.shape[1:] != (SIZE, SIZE)
            or not len(images) or not np.isfinite(images).all() or np.any(images <= 0)):
        raise ValueError("positive finite dense spatial priors required")
    result = images.reshape(-1, 16, PATCH, 16, PATCH).mean((2, 4)).reshape(-1, TOKENS)
    return (result/result.mean(1, keepdims=True)).astype(np.float32)


def control_permutation() -> np.ndarray:
    seed = int(object_digest({"spatial_control": "v1", "seed": SEED})[:16], 16)
    result = np.random.default_rng(seed).permutation(TOKENS)
    if np.array_equal(result, np.arange(TOKENS)):
        raise RuntimeError("identity spatial control prohibited")
    return result


class PatchFusion(nn.Module):
    def __init__(self, arm: str) -> None:
        super().__init__()
        if arm not in ARMS[1:]:
            raise ValueError("known non-reference fusion arm required")
        self.arm = arm
        self.gate = nn.Sequential(nn.Linear(DIMENSION, SETTINGS["gate_hidden"]), nn.Tanh(),
                                  nn.Linear(SETTINGS["gate_hidden"], 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.local_head = nn.Linear(DIMENSION, 2)
        nn.init.zeros_(self.local_head.weight)
        nn.init.zeros_(self.local_head.bias)
        if arm == "uniform":
            self.gate.requires_grad_(False)
        self.register_buffer("permutation", torch.from_numpy(control_permutation()))

    def weights(self, patches: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        if (patches.ndim != 3 or patches.shape[1:] != (TOKENS, DIMENSION)
                or priors.shape != patches.shape[:2] or not torch.isfinite(patches).all()
                or not torch.isfinite(priors).all() or (priors <= 0).any()):
            raise ValueError("finite image-only patch tokens and positive joined priors required")
        if self.arm == "uniform":
            return torch.full_like(priors, 1/TOKENS)
        values = SETTINGS["gate_logit_bound"]*self.gate(patches).squeeze(-1).tanh()
        if self.arm in ("anatomical", "permuted"):
            if self.arm == "permuted":
                priors = priors[:, self.permutation]
            priors = priors/priors.mean(1, keepdim=True)
            values = values+SETTINGS["prior_strength"]*priors.log()
        floor = SETTINGS["uniform_floor"]
        return floor/TOKENS+(1-floor)*values.softmax(-1)

    def forward(self, patches: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        return self.local_head((patches*self.weights(patches, priors)[..., None]).sum(1))


class SpatialReference(nn.Module):
    def __init__(self, reference: nn.Module, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError("known spatial arm required")
        self.arm, self.reference = arm, reference
        self.fusion = None if arm == "reference" else PatchFusion(arm)

    def forward(self, images: torch.Tensor, priors: torch.Tensor):
        if (images.ndim != 4 or images.shape[1:] != (3, SIZE, SIZE)
                or priors.shape != (len(images), TOKENS) or not torch.isfinite(priors).all()
                or (priors <= 0).any()):
            raise ValueError("RGB224 and corresponding positive patch priors required")
        if self.fusion is None:
            return self.reference(images)
        self.reference.weights = None
        features = self.reference.backbone.forward_features(images)
        patches = features["x_norm_patchtokens"]
        if self.reference.weights is None or patches.shape != (len(images), TOKENS, DIMENSION):
            raise RuntimeError("captured original reference attention and patch tokens required")
        global_logits = self.reference.global_head(features["x_norm_clstoken"])
        global_logits = global_logits+self.fusion(patches, priors)
        patch_logits = self.reference.patch_head(F.normalize(patches, dim=-1)*16)
        return global_logits, patch_logits, self.reference.weights


@dataclass
class SpatialBatch:
    images: np.ndarray
    labels: np.ndarray
    domains: np.ndarray
    patch_labels: np.ndarray
    priors: np.ndarray
    trace: dict[str, Any]


def paired_batch(
    rows: list[dict[str, Any]], load_rgb: Callable, load_prior: Callable, augmenter: Any,
    source: SourceModules, seed: int, epoch: int, step: int, *, attempts: int = 32,
    fas_probability: float = .2, pda_probability: float = .2,
) -> SpatialBatch:
    if (type(attempts) is not int or not 1 <= attempts <= 32
            or not 0 <= fas_probability <= 1 or not 0 <= pda_probability <= 1):
        raise ValueError("bounded attempts and valid augmentation probabilities required")
    sources = sorted({row["dataset"] for row in rows})
    if len(sources) != 3:
        raise ValueError("exactly three fitting sources required")
    strata = [[i for i, row in enumerate(rows)
               if row["dataset"] == domain and (row["binary_label"] == "attack") == bool(label)]
              for domain in sources for label in (0, 1)]
    if any(not pool for pool in strata):
        raise ValueError("complete source/class support required")
    # Mirror the pinned sampler because it has no spatial-transform callback.
    # Tests require pixel/label/PDA trace parity with the original implementation.
    for attempt in range(attempts):
        rng = source.augmentation.rng_for(seed, epoch, step, attempt)
        slots = list(range(6))*5+[(2*step) % 6, (2*step+1) % 6]
        selected = [int(rng.choice(strata[slot])) for slot in slots]
        images, labels, families, dense, transforms = [], [], [], [], []
        domains = np.array([slot//2 for slot in slots], dtype=np.int64)
        for index in selected:
            row = rows[index]
            rgb, prior = load_rgb(row), load_prior(row)
            if (rgb.shape != (SIZE, SIZE, 3) or rgb.dtype != np.uint8
                    or prior.shape != (SIZE, SIZE) or prior.dtype != np.float32
                    or not np.isfinite(prior).all() or np.any(prior <= 0)):
                raise ValueError("joined uint8 RGB224 and finite float32 spatial prior required")
            image, field = Image.fromarray(rgb), Image.fromarray(prior)
            flip = bool(rng.random() < .5)
            if flip:
                image, field = (item.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                                for item in (image, field))
            angle = float(rng.uniform(-10, 10))
            image = image.rotate(angle, Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
            field = field.rotate(angle, Image.Resampling.BILINEAR, fillcolor=1.)
            brightness = float(rng.uniform(.8, 1.2))
            image = ImageEnhance.Brightness(image).enhance(brightness)
            rgb = np.asarray(image).copy()
            label, family = int(row["binary_label"] == "attack"), "none"
            if rng.random() < fas_probability:
                rgb, label, family = augmenter.apply(rgb, label, rng)
            images.append(rgb)
            labels.append(label)
            families.append(family)
            dense.append(np.asarray(field).copy())
            transforms.append({"flip": flip, "angle": angle, "brightness": brightness})
        labels = np.asarray(labels, dtype=np.int64)
        counts = [[int(np.sum((domains == d) & (labels == y))) for y in (0, 1)] for d in range(3)]
        if any(min(pair) == 0 for pair in counts):
            continue
        images, dense = np.stack(images), np.stack(dense)
        donors, donor_priors = images.copy(), dense.copy()
        patches = np.repeat(labels[:, None], TOKENS, axis=1)
        donor_ids, donor_slots = [-1]*32, [-1]*32
        masks = np.zeros((32, TOKENS), dtype=bool)
        for i in range(32):
            if labels[i] and rng.random() < pda_probability:
                donor = int(rng.choice(np.flatnonzero((domains == domains[i]) & (labels == 0))))
                mask = rng.random((16, 16)) < .5
                pixel_mask = mask.repeat(PATCH, 0).repeat(PATCH, 1)
                images[i][pixel_mask] = donors[donor][pixel_mask]
                dense[i][pixel_mask] = donor_priors[donor][pixel_mask]
                patches[i][mask.ravel()] = 0
                donor_ids[i], donor_slots[i], masks[i] = selected[donor], donor, mask.ravel()
        trace = {"selected": selected, "families": families, "donors": donor_ids,
                 "mask_sha256": hashlib.sha256(masks.tobytes()).hexdigest(),
                 "redraws": attempt, "post_augmentation_counts": counts}
        trace["batch_sha256"] = hashlib.sha256(
            images.tobytes()+labels.tobytes()+domains.tobytes()+patches.tobytes()).hexdigest()
        priors = patch_prior(dense)
        trace.update({"spatial_transforms": transforms, "spatial_donor_slots": donor_slots,
                      "spatial_prior_sha256": hashlib.sha256(priors.tobytes()).hexdigest()})
        return SpatialBatch(images, labels, domains, patches, priors, trace)
    raise RuntimeError(f"post-augmentation source/class support exhausted after {attempts} attempts")


def optimizer_for(model: SpatialReference) -> torch.optim.Optimizer:
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = SETTINGS["backbone_lr"] if name.startswith("reference.backbone.") else SETTINGS["head_lr"]
        decay = SETTINGS["weight_decay"] if parameter.ndim > 1 and not name.endswith(".bias") else 0.
        groups.setdefault((lr, decay), []).append(parameter)
    return torch.optim.AdamW([{"params": params, "lr": lr, "weight_decay": decay}
                              for (lr, decay), params in groups.items()], betas=(.9, .999), eps=1e-8)


def check_gradients(model: SpatialReference, source: SourceModules) -> None:
    source.model.check_gradients(model.reference)
    if model.fusion is not None:
        for name, parameter in model.fusion.named_parameters():
            if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
                raise ValueError(f"missing/nonfinite spatial gradient: {name}")


@torch.no_grad()
def predict(model: SpatialReference, rows: list[dict[str, Any]], load_rgb: Callable,
            load_prior: Callable, source: SourceModules, device: str, guard: Callable) -> np.ndarray:
    model.eval()
    result = []
    for offset in range(0, len(rows), SETTINGS["batch"]):
        guard()
        group = rows[offset:offset+SETTINGS["batch"]]
        images = np.stack([load_rgb(row) for row in group])
        priors = patch_prior(np.stack([load_prior(row) for row in group]))
        outputs = model(source.model.tensor_images(images, device), torch.from_numpy(priors).to(device))
        values = outputs[0][:, 1]-outputs[0][:, 0]
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite spatial prediction")
        result.extend(values.cpu().double().tolist())
    return np.asarray(result, dtype=np.float64)


def validation_loss(logits: np.ndarray, rows: list[dict[str, Any]]) -> float:
    if logits.shape != (len(rows),) or not len(rows) or not np.isfinite(logits).all():
        raise ValueError("complete nonempty finite source-validation logits required")
    labels = np.asarray([row["binary_label"] == "attack" for row in rows], dtype=int)
    values, terms = np.logaddexp(0., logits)-labels*logits, []
    for domain in sorted({row["dataset"] for row in rows}):
        for label in (0, 1):
            mask = np.array([row["dataset"] == domain for row in rows]) & (labels == label)
            if not mask.any():
                raise ValueError("source validation stratum missing")
            terms.append(float(values[mask].mean()))
    return float(np.mean(terms))


def validate_state(state: dict[str, Any], fitting_count: int, arm: str) -> None:
    history = state["history"]
    values = [item["validation"] for item in history]
    if (state["arm"] != arm or state["seed"] != SEED
            or state["settings_sha256"] != object_digest(SETTINGS)
            or not history or len(history) > SETTINGS["epochs"] or not np.isfinite(values).all()
            or [item["epoch"] for item in history] != list(range(1, len(history)+1))
            or state["optimizer_steps"] != len(history)*((fitting_count+31)//32)
            or state["selected_epoch"] != int(np.argmin(values))+1
            or state["validation_loss"] != min(values)):
        raise ValueError("spatial checkpoint history/recipe/selection/step inventory mismatch")
    best, stale = float("inf"), 0
    for i, value in enumerate(values):
        stale = 0 if value < best else stale+1
        best = min(best, value)
        if stale >= SETTINGS["patience"] and i != len(values)-1:
            raise ValueError("spatial training continued beyond patience")
    if len(history) < SETTINGS["epochs"] and stale != SETTINGS["patience"]:
        raise ValueError("shortened spatial recipe is not successful early stopping")


def fit(
    model: SpatialReference, fitting: list[dict[str, Any]], validation: list[dict[str, Any]],
    load_rgb: Callable, load_prior: Callable, source: SourceModules, seed: int, device: str,
    guard: Callable, trace_path: Path,
) -> dict[str, Any]:
    optimizer = optimizer_for(model)
    augmenter = source.augmentation.FASAug()
    steps = (len(fitting)+31)//32
    if not steps:
        raise ValueError("nonempty spatial fitting population required")
    best_loss, selected, stale, best = float("inf"), 0, 0, None
    history = []
    with trace_path.open("x") as stream:
        import json
        for epoch in range(1, SETTINGS["epochs"]+1):
            model.train()
            totals = {name: 0. for name in ("global", "APL", "hardest_attack", "total")}
            for step in range(steps):
                guard()
                batch = paired_batch(fitting, load_rgb, load_prior, augmenter, source, seed, epoch, step)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(source.model.tensor_images(batch.images, device),
                                torch.from_numpy(batch.priors).to(device))
                values = source.model.losses(outputs, torch.from_numpy(batch.labels).to(device),
                                             torch.from_numpy(batch.domains).to(device),
                                             torch.from_numpy(batch.patch_labels).to(device), tail=False)
                values["total"].backward()
                check_gradients(model, source)
                optimizer.step()
                for name, value in values.items():
                    totals[name] += value.item()
                stream.write(json.dumps({"epoch": epoch, "step": step, **batch.trace}, sort_keys=True)+"\n")
                stream.flush()
                guard()
            score = validation_loss(predict(model, validation, load_rgb, load_prior, source, device, guard),
                                    validation)
            history.append({"epoch": epoch, "validation": score,
                            "mean_training_losses": {name: value/steps for name, value in totals.items()}})
            if score < best_loss:
                best_loss, selected, stale = score, epoch, 0
                best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            print("SPATIAL_DINO_EPOCH", model.arm, epoch, score, flush=True)
            if stale >= SETTINGS["patience"]:
                break
    if best is None:
        raise RuntimeError("no finite source-selected spatial checkpoint")
    model.load_state_dict(best, strict=True)
    del optimizer
    gc.collect()
    return {"state_dict": best, "history": history, "selected_epoch": selected,
            "validation_loss": best_loss, "optimizer_steps": len(history)*steps,
            "arm": model.arm, "seed": seed, "settings_sha256": object_digest(SETTINGS),
            "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest()}
