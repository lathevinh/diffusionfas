"""Matched frozen-adapted-DINO clip heads and addressed order controls."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import copy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import addressed_seed, window_indices

SEED = 20260917
FRAMES = 16
DIMENSION = 768
ARMS = ("single", "mean", "temporal", "shuffled")
DIAGNOSTIC = "temporal_inference_shuffle"
SYSTEMS = (*ARMS, DIAGNOSTIC)
CONTRASTS = (
    ("mean_minus_single", "mean", "single"),
    ("temporal_minus_mean", "temporal", "mean"),
    ("shuffled_minus_mean", "shuffled", "mean"),
    ("temporal_minus_shuffled", "temporal", "shuffled"),
    ("temporal_minus_single", "temporal", "single"),
    ("inference_shuffle_minus_temporal", DIAGNOSTIC, "temporal"),
)
SETTINGS = {"epochs": 200, "lr": .001, "weight_decay": .05, "hidden": 32,
            "dtype": "float64", "training_device": "cpu", "full_batch": True,
            "seed": SEED, "early_stopping": False,
            "selection": "earliest_strict_best_source_equal_domain_equal_class_BCE"}


def middle_window(timestamps: Sequence[float]) -> tuple[int, ...]:
    times = tuple(timestamps)
    if len(times) < 2:
        raise ValueError("insufficient_duration")
    gaps = np.diff(times)
    if not np.isfinite(times).all() or np.any(gaps <= 0):
        raise ValueError("strict finite native timestamps required")
    end = times[-1]+float(np.median(gaps))
    if end-times[0]+1e-8 < 2:
        raise ValueError("insufficient_duration")
    return window_indices(times, (times[0]+end-2)/2, 2, 8)


def shuffle_indices(keys: Sequence[str], target: str) -> np.ndarray:
    if not keys or len(set(keys)) != len(keys) or not target:
        raise ValueError("nonempty unique original keys and target required")
    result = []
    for key in keys:
        permutation = np.random.default_rng(addressed_seed(SEED, "clip-order-v1", target, key)).permutation(FRAMES)
        if np.array_equal(permutation, np.arange(FRAMES)):
            permutation = np.roll(permutation, 1)
        result.append(permutation)
    return np.asarray(result)


def shuffled_features(features: np.ndarray, keys: Sequence[str], target: str) -> np.ndarray:
    if features.shape != (len(keys), FRAMES, DIMENSION) or not np.isfinite(features).all():
        raise ValueError("finite joined N,16,768 features required")
    return features[np.arange(len(keys))[:, None], shuffle_indices(keys, target)]


class ClipHead(nn.Module):
    def __init__(self, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError("declared trainable clip arm required")
        self.arm = arm
        if arm in ("temporal", "shuffled"):
            self.projection = nn.Linear(DIMENSION, SETTINGS["hidden"])
            self.recurrent = nn.GRU(SETTINGS["hidden"], SETTINGS["hidden"], batch_first=True)
            self.classifier = nn.Linear(DIMENSION+SETTINGS["hidden"], 1)
        else:
            self.classifier = nn.Linear(DIMENSION, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != (FRAMES, DIMENSION):
            raise ValueError("N,16,768 frozen CLS features required")
        mean = features.mean(dim=1)
        if self.arm == "single":
            value = features[:, FRAMES//2]
        elif self.arm == "mean":
            value = mean
        else:
            residual = features-mean[:, None]
            _, state = self.recurrent(torch.tanh(self.projection(residual)))
            value = torch.cat((mean, state[-1]), dim=1)
        return self.classifier(value).squeeze(-1)


def initial_head(arm: str) -> ClipHead:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        return ClipHead(arm).double()


def balanced_weights(labels: np.ndarray, domains: Sequence[str]) -> np.ndarray:
    if (labels.ndim != 1 or len(labels) != len(domains) or set(labels.tolist()) != {0, 1}
            or not domains):
        raise ValueError("joined binary labels and nonempty domains required")
    names = sorted(set(domains))
    domain = np.asarray(domains)
    weights = np.empty(len(labels), dtype=np.float64)
    for name in names:
        for label in (0, 1):
            mask = (domain == name) & (labels == label)
            if not mask.any():
                raise ValueError("both classes required in each source")
            weights[mask] = 1/(2*len(names)*mask.sum())
    return weights


def fit_head(arm: str, features: np.ndarray, labels: np.ndarray, domains: Sequence[str],
             validation: np.ndarray, validation_labels: np.ndarray,
             validation_domains: Sequence[str], guard: Callable[[], None], *,
             epochs: int = SETTINGS["epochs"]) -> tuple[ClipHead, dict[str, Any]]:
    if not 1 <= epochs <= SETTINGS["epochs"]:
        raise ValueError("bounded positive head epochs required")
    arrays = (features, validation)
    if any(x.shape != (len(y), FRAMES, DIMENSION) or not np.isfinite(x).all()
           for x, y in zip(arrays, (labels, validation_labels), strict=True)):
        raise ValueError("finite joined fitting and validation clips required")
    model = initial_head(arm)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.ndim > 1],
         "weight_decay": SETTINGS["weight_decay"]},
        {"params": [p for p in model.parameters() if p.ndim == 1], "weight_decay": 0.},
    ], lr=SETTINGS["lr"])
    x, v = (torch.from_numpy(np.asarray(value, dtype=np.float64)) for value in arrays)
    y, vy = (torch.from_numpy(value.astype(np.float64)) for value in (labels, validation_labels))
    w = torch.from_numpy(balanced_weights(labels, domains))
    vw = torch.from_numpy(balanced_weights(validation_labels, validation_domains))
    best_loss, best, selected, history = float("inf"), None, 0, []
    for epoch in range(1, epochs+1):
        guard()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = w @ F.binary_cross_entropy_with_logits(model(x), y, reduction="none")
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite clip fitting loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite or missing clip-head gradients")
        optimizer.step()
        guard()
        model.eval()
        with torch.no_grad():
            value = float(vw @ F.binary_cross_entropy_with_logits(model(v), vy, reduction="none"))
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite source-validation loss")
        history.append({"epoch": epoch, "fitting_BCE": float(loss.detach()), "validation_BCE": value})
        if value < best_loss:
            best_loss, best, selected = value, copy.deepcopy(model.state_dict()), epoch
    if best is None:
        raise RuntimeError("no finite source-selected clip head")
    model.load_state_dict(best, strict=True)
    model.eval()
    return model, {"arm": arm, "seed": SEED, "settings": SETTINGS,
                   "selected_epoch": selected, "validation_BCE": best_loss, "history": history,
                   "initial_state": initial, "state_dict": best}


def logits(model: ClipHead, features: np.ndarray, guard: Callable[[], None]) -> np.ndarray:
    if (features.ndim != 3 or features.shape[1:] != (FRAMES, DIMENSION)
            or not len(features) or not np.isfinite(features).all()):
        raise ValueError("nonempty finite clip features required")
    values = []
    with torch.inference_mode():
        for start in range(0, len(features), 64):
            guard()
            values.append(model(torch.from_numpy(features[start:start+64].astype(np.float64))).numpy())
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite clip logits")
    return result
