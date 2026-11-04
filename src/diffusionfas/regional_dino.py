"""Frozen regional DINO features and source-balanced group-DRO linear heads."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .mat_input import POLICY as MAT_POLICY

REPRESENTATIONS = ("global", "anatomical", "spatial_control")
OBJECTIVES = ("erm", "group_dro")
ARMS = tuple(f"{representation}__{objective}"
             for representation in REPRESENTATIONS for objective in OBJECTIVES)
CONTRASTS = tuple(("minus_global_erm", arm, ARMS[0]) for arm in ARMS[1:]) + (
    ("anatomical_minus_spatial_erm", "anatomical__erm", "spatial_control__erm"),
    ("anatomical_minus_spatial_dro", "anatomical__group_dro", "spatial_control__group_dro"),
    ("dro_minus_erm_anatomical", "anatomical__group_dro", "anatomical__erm"),
    ("dro_minus_erm_spatial", "spatial_control__group_dro", "spatial_control__erm"),
)
RESOLUTION = 518
PATCH_SIZE = 14
DIMENSIONS = {"global": 1536, "anatomical": 4608, "spatial_control": 4608}
HEAD_SETTINGS = {"epochs": 200, "learning_rate": .001, "weight_decay": .05,
                 "dro_step_size": .1}
REGIONS = ("left_eye", "right_eye", "nose", "mouth")


class RegionSupportError(ValueError):
    """A retained shared technical failure, never a replacement regional feature."""


def crop_landmarks(transaction: dict[str, Any]) -> np.ndarray:
    """Transfer pixel centers through the saved rotation, crop and OpenCV resize."""
    points = np.asarray(transaction["landmarks"], dtype=np.float64)
    geometry = transaction["geometry"]
    matrix = np.asarray(geometry["alignment_matrix"], dtype=np.float64)
    bounds = np.asarray(geometry["crop_xyxy_half_open"], dtype=np.float64)
    shape = transaction["crop_shape"]
    if (points.shape != (5, 2) or matrix.shape != (2, 3) or bounds.shape != (4,)
            or not all(np.isfinite(value).all() for value in (points, matrix, bounds))
            or np.any(bounds[2:] <= bounds[:2]) or np.any(bounds != np.floor(bounds))
            or list(shape) != [int(bounds[3]-bounds[1]), int(bounds[2]-bounds[0]), 3]
            or geometry["context_scale"] != 1.30
            or geometry["interpolation"] != "bicubic" or geometry["border"] != "constant_zero"):
        raise ValueError("exact finite historical shared-context crop geometry required")
    rotated = points @ matrix[:, :2].T + matrix[:, 2]
    return (rotated - bounds[:2] + .5) * RESOLUTION / (bounds[2:] - bounds[:2]) - .5


def region_weights(points: np.ndarray) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("five finite landmarks in historical RGB518 coordinates required")
    eye = points[1] - points[0]
    distance = float(np.linalg.norm(eye))
    if distance <= 1e-8:
        raise RegionSupportError("degenerate regional DINO eye geometry")
    horizontal = eye / distance
    vertical = np.array([-horizontal[1], horizontal[0]])
    grid = RESOLUTION // PATCH_SIZE
    yy, xx = np.indices((grid, grid))
    centers = np.column_stack(((xx.ravel()+.5)*PATCH_SIZE-.5,
                               (yy.ravel()+.5)*PATCH_SIZE-.5))
    landmarks = [points[0], points[1], points[2], points[3:5].mean(axis=0)]
    anchors = np.asarray([[.25, .25], [.75, .25], [.25, .75], [.75, .75]]) * RESOLUTION - .5
    anatomical, control = [], []
    for name, center, anchor in zip(REGIONS, landmarks, anchors, strict=True):
        radii = np.asarray(MAT_POLICY["ellipse_radii_in_eye_distances"][name]) * distance
        extent = np.sqrt((radii[0]*horizontal)**2 + (radii[1]*vertical)**2)
        if np.any(center-extent < 0) or np.any(center+extent > RESOLUTION-1):
            raise RegionSupportError(f"regional DINO {name} ellipse outside historical crop")
        delta = centers-center
        selected = ((delta @ horizontal / radii[0])**2
                    + (delta @ vertical / radii[1])**2 <= 1)
        count = int(selected.sum())
        if not 0 < count < grid*grid:
            raise RegionSupportError(f"regional DINO {name} has no valid patch-center support")
        other = np.zeros(grid*grid, dtype=bool)
        order = np.argsort(np.square(centers-anchor).sum(axis=1), kind="stable")
        other[order[:count]] = True
        anatomical.append(selected.astype(np.float32)/count)
        control.append(other.astype(np.float32)/count)
    return {"anatomical": np.stack(anatomical), "spatial_control": np.stack(control)}


def pool_tokens(cls: np.ndarray, patches: np.ndarray,
                weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if (cls.shape != (1, 768) or patches.shape != (1, 1369, 768)
            or cls.dtype != np.float32 or patches.dtype != np.float32
            or not np.isfinite(cls).all() or not np.isfinite(patches).all()
            or set(weights) != {"anatomical", "spatial_control"}):
        raise ValueError("native518 finite normalized CLS and image-only patch tokens required")
    global_feature = np.concatenate((cls, patches.mean(axis=1)), axis=1)
    result = {"global": global_feature}
    for name in ("anatomical", "spatial_control"):
        value = weights[name]
        if (value.shape != (4, 1369) or value.dtype != np.float32
                or not np.isfinite(value).all() or np.any(value < 0)
                or not np.allclose(value.sum(axis=1), 1, rtol=0, atol=1e-6)):
            raise ValueError("four finite normalized nonnegative spatial pooling weights required")
        local = (value @ patches[0]).reshape(1, -1)
        result[name] = np.concatenate((global_feature, local), axis=1)
    return result


def group_matrix(labels: np.ndarray, domains: list[str]) -> tuple[list[str], np.ndarray]:
    if (labels.ndim != 1 or len(labels) != len(domains) or not len(labels)
            or not np.isin(labels, (0, 1)).all()
            or any(not isinstance(domain, str) or not domain for domain in domains)):
        raise ValueError("nonempty binary labels and exact named domain joins required")
    names = sorted(set(domains))
    weights = np.zeros((len(names), len(labels)), dtype=np.float64)
    for index, name in enumerate(names):
        for label in (0, 1):
            mask = (np.asarray(domains) == name) & (labels == label)
            if not mask.any():
                raise ValueError("both classes required in every source domain")
            weights[index, mask] = .5 / mask.sum()
    return names, weights


def dro_weights(log_weights: np.ndarray, losses: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    if (log_weights.ndim != 1 or not len(log_weights) or losses.shape != log_weights.shape
            or not np.isfinite(log_weights).all() or not np.isfinite(losses).all()
            or np.any(losses < 0) or not np.isfinite(step) or step < 0):
        raise ValueError("finite nonnegative group losses and DRO step required")
    updated = log_weights + step*losses
    if not np.isfinite(updated).all():
        raise FloatingPointError("nonfinite group-DRO adversary update")
    shifted = updated-updated.max()
    normalized = shifted-np.log(np.exp(shifted).sum())
    return normalized, np.exp(normalized)


def fit_group_dro(
    x: np.ndarray, y: np.ndarray, domains: list[str], validation_x: np.ndarray,
    validation_y: np.ndarray, validation_domains: list[str], *, seed: int,
    fitting: Any, guard: Callable[[], None], settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = HEAD_SETTINGS if settings is None else settings
    if (x.ndim != 2 or x.shape[0] != len(y) or not x.shape[1]
            or validation_x.ndim != 2 or validation_x.shape != (len(validation_y), x.shape[1])
            or not np.isfinite(x).all() or not np.isfinite(validation_x).all()
            or type(settings["epochs"]) is not int or settings["epochs"] < 1
            or not np.isfinite([settings["learning_rate"], settings["weight_decay"],
                                settings["dro_step_size"]]).all()
            or settings["learning_rate"] <= 0 or settings["weight_decay"] < 0
            or settings["learning_rate"]*settings["weight_decay"] >= 1
            or settings["dro_step_size"] < 0):
        raise ValueError("finite compatible source feature matrices and valid fixed head settings required")
    names, weights = group_matrix(y, domains)
    validation_names, validation_weights = group_matrix(validation_y, validation_domains)
    if names != validation_names:
        raise ValueError("identical fitting/validation source domains required")
    x, validation_x = x.astype(np.float64), validation_x.astype(np.float64)
    coefficients = np.r_[np.random.default_rng(seed).normal(0, .001, x.shape[1]), 0.]
    first, second = np.zeros_like(coefficients), np.zeros_like(coefficients)
    log_weights = np.full(len(names), -np.log(len(names)))
    best_loss, best_epoch, best = float("inf"), 0, None
    history, adversary_history = [], []
    for epoch in range(1, settings["epochs"]+1):
        guard()
        logits = x @ coefficients[:-1] + coefficients[-1]
        losses = weights @ (np.logaddexp(0, logits)-y*logits)
        log_weights, q = dro_weights(log_weights, losses, settings["dro_step_size"])
        residual = (q @ weights) * (fitting.sigmoid(logits)-y)
        gradient = np.r_[x.T @ residual, residual.sum()]
        first = .9*first+.1*gradient
        second = .999*second+.001*gradient*gradient
        update = (first/(1-.9**epoch))/(np.sqrt(second/(1-.999**epoch))+1e-8)
        coefficients[:-1] *= 1-settings["learning_rate"]*settings["weight_decay"]
        coefficients -= settings["learning_rate"]*update
        values = validation_x @ coefficients[:-1]+coefficients[-1]
        loss = float((validation_weights @ (np.logaddexp(0, values)-validation_y*values)).mean())
        if not np.isfinite(coefficients).all() or not np.isfinite(loss):
            raise FloatingPointError("nonfinite group-DRO head optimization")
        history.append(loss)
        adversary_history.append(q.tolist())
        if loss < best_loss:
            best_loss, best_epoch, best = loss, epoch, coefficients.copy()
    if best is None:
        raise RuntimeError("no selected group-DRO checkpoint")
    return {"coefficients": best.tolist(), "selected_epoch": best_epoch,
            "validation_bce": best_loss, "validation_history": history, "seed": seed,
            "optimizer": "full_batch_adamw_float64_bias_not_decayed",
            "group_domains": names, "adversary_history": adversary_history,
            "dro_step_size": settings["dro_step_size"],
            "checkpoint_criterion": "equal_source_equal_class_validation_BCE_not_worst_group"}
