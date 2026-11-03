"""Explicit local-media preparation; no extraction, downloads or target access."""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace
import hashlib
import importlib
from pathlib import Path
import sys
import time
from typing import Protocol
from collections.abc import Callable, Iterator

import cv2
import numpy as np

from .data import Video, window_indices
from .io import digest


class MediaFailure(ValueError):
    """A recorded technical failure, not a negative PAD prediction."""


class FaceProvider(Protocol):
    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None: ...

    def crop(self, rgb: np.ndarray, box: np.ndarray, landmarks: np.ndarray) -> np.ndarray: ...


class NativeReplay(Protocol):
    def __call__(self, path: Path, consume: Callable[[int, np.ndarray], None], timeout: float) -> None: ...


class FasFaceProvider:
    def __init__(
        self, source_root: Path, preprocessing_sha256: str,
        detector: Path, detector_sha256: str,
    ) -> None:
        module_path = source_root / "src/fas/preprocessing.py"
        if digest(module_path) != preprocessing_sha256 or digest(detector) != detector_sha256:
            raise ValueError("source preprocessing or detector hash mismatch")
        sys.path.insert(0, str(source_root / "src"))
        self.module = importlib.import_module("fas.preprocessing")
        if Path(self.module.__file__).resolve() != module_path.resolve():
            raise ValueError("imported FAS preprocessing is not the bound source module")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(detector), sess_options=options,
                                           providers=["CPUExecutionProvider"])

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            box, landmarks = self.module.detect_face(self.session, rgb)
        except ValueError as error:
            if str(error) != "no_face_above_frozen_threshold":
                raise
            return None
        return box[:4], landmarks

    def crop(self, rgb: np.ndarray, box: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        crop, _ = self.module.shared_context_crop(rgb, box, landmarks)
        return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)


def interpolate_track(
    detections: list[tuple[np.ndarray, np.ndarray] | None], *, maximum_gap: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if not detections or detections[0] is None or detections[-1] is None:
        raise MediaFailure("missing_track_endpoint")
    boxes, points = [], []
    for detection in detections:
        if detection is None:
            boxes.append(np.full(4, np.nan))
            points.append(np.full((5, 2), np.nan))
        else:
            box, landmarks = detection
            if box.shape != (4,) or landmarks.shape != (5, 2) or not np.isfinite(box).all() or not np.isfinite(landmarks).all():
                raise MediaFailure("invalid_detection_geometry")
            if np.any(box[2:] <= box[:2]):
                raise MediaFailure("nonpositive_face_box")
            boxes.append(box.astype(np.float64))
            points.append(landmarks.astype(np.float64))
    boxes_array, points_array = np.array(boxes), np.array(points)
    known = np.flatnonzero(np.isfinite(boxes_array).all(axis=1))
    for left, right in zip(known, known[1:]):
        if right - left - 1 > maximum_gap:
            raise MediaFailure("track_gap_exceeds_limit")
        for index in range(left + 1, right):
            weight = (index - left) / (right - left)
            boxes_array[index] = (1 - weight) * boxes_array[left] + weight * boxes_array[right]
            points_array[index] = (1 - weight) * points_array[left] + weight * points_array[right]
    for previous, current in zip(boxes_array, boxes_array[1:]):
        previous_center = (previous[:2] + previous[2:]) / 2
        current_center = (current[:2] + current[2:]) / 2
        if np.linalg.norm(current_center - previous_center) > np.linalg.norm(previous[2:] - previous[:2]) * 0.5:
            raise MediaFailure("ambiguous_track_jump")
        ratio = np.prod(current[2:] - current[:2]) / np.prod(previous[2:] - previous[:2])
        if not 0.5 <= ratio <= 2:
            raise MediaFailure("ambiguous_track_scale")
    # Centered smoothing reduces detector jitter without nonlinear face warping.
    for values in (boxes_array, points_array):
        source = values.copy()
        for index in range(len(values)):
            values[index] = source[max(0, index - 1):index + 2].mean(axis=0)
    return boxes_array, points_array


def prepare_video(
    video: Video, provider: FaceProvider, output: Path, *,
    fps: int = 8, max_seconds: float = 180, max_output_bytes: int = 512 * 1024**2,
    window_start: float | None = None, window_duration: float | None = None,
    native_replay: NativeReplay | None = None,
) -> Video:
    if not video.media_path or not video.media_sha256 or not video.frame_hashes:
        raise ValueError("bound extracted local media path, hash and native frame hashes required")
    if video.status != "eligible_pending_tracking":
        raise ValueError("only decode-eligible original can be prepared")
    path = Path(video.media_path)
    if digest(path) != video.media_sha256:
        raise ValueError("local media bytes do not match accepted original")
    interval = video.interval
    if interval is None or interval[1] - interval[0] + 1e-8 < 2:
        raise MediaFailure("insufficient_duration")
    if (window_start is None) != (window_duration is None):
        raise ValueError("window start and duration must be supplied together")
    if window_start is None:
        start = interval[0]
        duration = int((interval[1] - interval[0] + 1e-8) * fps) / fps
    else:
        start, duration = window_start, window_duration
        if duration is None or start < interval[0] - 1e-8 or start + duration > interval[1] + 1e-8:
            raise ValueError("requested fixed window outside accepted native interval")
    indexes = window_indices(video.timestamps, start, duration, fps)
    maximum_cache_bytes = len(indexes) * 256 * 256 * 3 + len(indexes) * (8 + 4 * 8 + 10 * 8) + 16384
    if maximum_cache_bytes > max_output_bytes:
        raise MediaFailure("cache_output_budget_exceeded")
    wanted, deadline = set(indexes), time.monotonic() + max_seconds

    def selected_frames() -> Iterator[np.ndarray]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise MediaFailure("local_video_open_failed")
        order, selected = 0, 0
        try:
            while True:
                if time.monotonic() > deadline:
                    raise MediaFailure("video_preparation_timeout")
                success, bgr = capture.read()
                if not success:
                    break
                if order >= len(video.timestamps):
                    raise MediaFailure("native_frame_count_mismatch")
                if order in wanted:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if hashlib.sha256(rgb.tobytes()).hexdigest() != video.frame_hashes[order]:
                        raise MediaFailure("native_RGB_hash_mismatch_no_decoder_waiver")
                    selected += 1
                    yield rgb
                order += 1
        finally:
            capture.release()
        if order != len(video.timestamps) or selected != len(indexes):
            raise MediaFailure("native_frame_count_mismatch")

    def consume_selected(consume: Callable[[np.ndarray], None]) -> None:
        if native_replay is None:
            with closing(selected_frames()) as frames:
                for frame in frames:
                    consume(frame)
            return
        order, selected = 0, 0

        def verify(index: int, rgb: np.ndarray) -> None:
            nonlocal order, selected
            if time.monotonic() > deadline:
                raise MediaFailure("video_preparation_timeout")
            if index != order or order >= len(video.timestamps):
                raise MediaFailure("native_frame_count_mismatch")
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise MediaFailure("native_RGB_shape_mismatch")
            if hashlib.sha256(rgb.tobytes()).hexdigest() != video.frame_hashes[order]:
                raise MediaFailure("native_RGB_hash_mismatch_no_decoder_waiver")
            order += 1
            if index in wanted:
                selected += 1
                consume(rgb)

        native_replay(path, verify, max(0.001, deadline - time.monotonic()))
        if order != len(video.timestamps) or selected != len(indexes):
            raise MediaFailure("native_frame_count_mismatch")

    detections = []
    consume_selected(lambda frame: detections.append(provider.detect(frame)))
    boxes, landmarks = interpolate_track(detections)
    crops = []

    def crop_frame(frame: np.ndarray) -> None:
        slot = len(crops)
        crop = provider.crop(frame, boxes[slot], landmarks[slot])
        if crop.shape != (256, 256, 3) or crop.dtype != np.uint8:
            raise ValueError("face provider must return RGB256 uint8")
        crops.append(crop)

    consume_selected(crop_frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez(stream, frames=np.stack(crops),
                 timestamps=start + (np.arange(len(indexes)) + 0.5) / fps,
                 native_timestamps=np.array([video.timestamps[i] for i in indexes]),
                 boxes=boxes, landmarks=landmarks, native_orders=np.array(indexes))
    if output.stat().st_size > max_output_bytes:
        raise OSError("written cache exceeds predeclared byte budget")
    return replace(video, status="prepared", cache_path=str(output.resolve()), cache_sha256=digest(output))
