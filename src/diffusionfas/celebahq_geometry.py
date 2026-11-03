"""CelebA-HQ-style square geometry, not the dataset's enhancement pipeline."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .preprocessing import MediaFailure

POLICY = {
    "name": "celebahq_geometry_v1",
    "output_resolution": 256,
    "render_resolution": 1024,
    "eye_distance_half_side": 2.0,
    "eye_mouth_distance_half_side": 1.8,
    "center_eye_mouth_offset": 0.1,
    "border": "reflect_101",
    "interpolation": "bilinear_1024_then_area_256",
    "enhancement_superresolution_or_beautification": False,
    "bit_exact_celebahq_claimed": False,
    "reference": "https://github.com/tkarras/progressive_growing_of_gans/blob/master/dataset_tool.py",
}

REAL_PIXEL_POLICY = {
    **POLICY,
    "name": "celebahq_real_pixels_v1",
    "border": "forbidden_verified_bilinear_support",
    "quad_bounds": "closed_native_pixel_center_rectangle",
    "crop_shrinking_allowed": False,
    "crop_shifting_allowed": False,
}


def square_geometry(landmarks: np.ndarray, resolution: int = 256) -> dict[str, Any]:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise MediaFailure("celebahq_invalid_landmarks")
    if type(resolution) is not int or resolution < 1:
        raise ValueError("positive integral output resolution required")
    eyes = points[:2].mean(axis=0)
    eye_delta = points[1] - points[0]
    mouth_delta = points[3:5].mean(axis=0) - eyes
    eye_distance, mouth_distance = np.linalg.norm(eye_delta), np.linalg.norm(mouth_delta)
    horizontal = eye_delta + np.array([mouth_delta[1], -mouth_delta[0]])
    length = np.linalg.norm(horizontal)
    if min(eye_distance, mouth_distance, length) <= 1e-8:
        raise MediaFailure("celebahq_degenerate_landmarks")
    horizontal /= length
    vertical = np.array([-horizontal[1], horizontal[0]])
    half_side = max(2.0 * eye_distance, 1.8 * mouth_distance)
    center = eyes + 0.1 * mouth_delta
    basis = np.stack((horizontal, vertical))
    linear = basis * (resolution / (2 * half_side))
    translation = (resolution - 1) / 2 - linear @ center
    matrix = np.column_stack((linear, translation))
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    quad = center + half_side * (corners @ basis)
    return {"source_to_output_matrix": matrix.tolist(), "quad": quad.tolist(),
            "center": center.tolist(), "side_source_pixels": float(2 * half_side),
            "eye_distance_output_pixels": float(eye_distance * resolution / (2 * half_side))}


def crop_celebahq(rgb: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
        raise ValueError("nonempty uint8 RGB image at least 2x2 required")
    geometry = square_geometry(landmarks)
    render = POLICY["render_resolution"]
    matrix = np.asarray(square_geometry(landmarks, render)["source_to_output_matrix"])
    aligned = cv2.warpAffine(rgb, matrix, (render, render), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    valid = cv2.warpAffine(np.ones(rgb.shape[:2], dtype=np.float32), matrix, (render, render),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    output = cv2.resize(aligned, (256, 256), interpolation=cv2.INTER_AREA)
    return output, {**geometry, "padding_fraction": float(np.clip(1 - valid.mean(), 0, 1)),
                    "policy": POLICY["name"]}


def crop_real_pixels(rgb: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
        raise ValueError("nonempty uint8 RGB image at least 2x2 required")
    geometry = square_geometry(landmarks)
    quad = np.asarray(geometry["quad"])
    height, width = rgb.shape[:2]
    margin = float(np.min(np.stack((quad[:, 0], quad[:, 1],
                                    width - 1 - quad[:, 0], height - 1 - quad[:, 1]))))
    if margin < 0:
        raise MediaFailure("celebahq_square_outside_real_pixels")
    render = REAL_PIXEL_POLICY["render_resolution"]
    matrix = np.asarray(square_geometry(landmarks, render)["source_to_output_matrix"])
    # Verify the actual quantized interpolation footprint before sampling RGB.
    support = cv2.warpAffine(np.ones((height, width), dtype=np.float32), matrix, (render, render),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if not np.all(support == 1):
        raise MediaFailure("celebahq_bilinear_support_outside_real_pixels")
    aligned = cv2.warpAffine(rgb, matrix, (render, render), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    output = cv2.resize(aligned, (256, 256), interpolation=cv2.INTER_AREA)
    return output, {**geometry, "padding_fraction": 0.0, "bilinear_support_verified": True,
                    "minimum_source_margin_pixels": margin, "policy": REAL_PIXEL_POLICY["name"]}
