"""MAT native-512 real-pixel crops, binary known masks and masked-only errors."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from .celebahq_geometry import square_geometry
from .crop_fit_audit import quad_margin
from .preprocessing import MediaFailure

POLICY = {
    "name": "mat_celebahq512_real_pixels_v1", "resolution": 512, "render_resolution": 1024,
    "geometry": "unchanged_celebahq_eye_mouth_square",
    "source": "native_RGB_not_resized_256_cache", "padding": False, "shift_or_shrink": False,
    "mask": "float32_1_known_0_hole", "normalization": "RGB_float32_div_127.5_minus_1",
    "mask_regions": ["left_eye", "right_eye", "nose", "mouth"],
    "ellipse_radii_in_eye_distances": {
        "left_eye": [.25, .18], "right_eye": [.25, .18], "nose": [.23, .30], "mouth": [.42, .23]},
    "noise_mode": "const", "truncation_psi": 1.0,
    "internal_dropout_seeded_even_in_eval": True, "draws_per_mask": 1,
    "bit_exact_training_crop_claimed": False, "human_review_required": True,
}


def crop_native512(rgb: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
        raise ValueError("native uint8 RGB image at least 2x2 required")
    height, width = rgb.shape[:2]
    margin = quad_margin(landmarks, width, height)
    if margin < 0:
        raise MediaFailure("mat_square_outside_real_pixels")
    geometry = square_geometry(landmarks, 512)
    matrix = np.asarray(square_geometry(landmarks, 1024)["source_to_output_matrix"])
    support = cv2.warpAffine(np.ones((height, width), dtype=np.float32), matrix, (1024, 1024),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if not np.all(support == 1):
        raise MediaFailure("mat_bilinear_support_outside_real_pixels")
    aligned = cv2.warpAffine(rgb, matrix, (1024, 1024), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cropped = cv2.resize(aligned, (512, 512), interpolation=cv2.INTER_AREA)
    linear = np.asarray(geometry["source_to_output_matrix"])
    points = np.asarray(landmarks) @ linear[:, :2].T + linear[:, 2]
    return cropped, {**geometry, "output_landmarks": points.tolist(), "padding_fraction": 0.0,
                     "bilinear_support_verified": True, "minimum_source_margin_pixels": margin,
                     "native_width_height": [width, height], "policy": POLICY["name"]}


def anatomical_masks(points: np.ndarray) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("five finite output landmarks required")
    eye = points[1] - points[0]
    distance = float(np.linalg.norm(eye))
    if distance <= 1e-8:
        raise MediaFailure("mat_degenerate_output_landmarks")
    horizontal, vertical = eye / distance, np.array([-eye[1], eye[0]]) / distance
    centers = {"left_eye": points[0], "right_eye": points[1], "nose": points[2],
               "mouth": points[3:5].mean(axis=0)}
    yy, xx = np.indices((512, 512))
    masks = {}
    for name in POLICY["mask_regions"]:
        rx, ry = np.asarray(POLICY["ellipse_radii_in_eye_distances"][name]) * distance
        center = centers[name]
        extent = np.sqrt((rx * horizontal) ** 2 + (ry * vertical) ** 2)
        if np.any(center - extent < 0) or np.any(center + extent > 511):
            raise MediaFailure("mat_anatomical_mask_outside_crop")
        dx, dy = xx - center[0], yy - center[1]
        x = dx * horizontal[0] + dy * horizontal[1]
        y = dx * vertical[0] + dy * vertical[1]
        hole = (x / rx) ** 2 + (y / ry) ** 2 <= 1
        if not hole.any() or hole.all():
            raise MediaFailure("mat_empty_or_full_hole")
        masks[name] = (~hole).astype(np.float32)
    return masks


def checked_image_mask(rgb: np.ndarray, known: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.shape != (512, 512, 3):
        raise ValueError("uint8 RGB512 required")
    if (known.dtype != np.float32 or known.shape != (512, 512)
            or not np.isin(known, [0., 1.]).all() or known.min() != 0 or known.max() != 1):
        raise ValueError("binary float32 mask with both known and hole pixels required")


def model_input(rgb: np.ndarray, known: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    checked_image_mask(rgb, known)
    mask = torch.from_numpy(known.copy())[None, None]
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().div(127.5).sub(1)
    return image * mask, mask


def masked_error(rgb: np.ndarray, prediction: np.ndarray, known: np.ndarray) -> tuple[float, np.ndarray]:
    checked_image_mask(rgb, known)
    if prediction.shape != (512, 512, 3) or not np.isfinite(prediction).all():
        raise ValueError("finite RGB512 float prediction required")
    target = rgb.astype(np.float64) / 127.5 - 1
    errors = np.square(prediction.astype(np.float64) - target).mean(axis=2)
    hole = known == 0
    return float(errors[hole].mean()), np.where(hole, errors, 0)
