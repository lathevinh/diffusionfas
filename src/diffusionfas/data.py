"""Partition-preserving records and timestamp-addressed clip sampling."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
from typing import Any

from .io import object_digest, read_json

DOMAINS = ("OULU-NPU", "CASIA-FASD", "MSU-MFSD", "SiW-Mv2")
PARTITIONS = ("fitting", "validation", "calibration", "evaluation")


@dataclass(frozen=True)
class Video:
    key: str
    dataset: str
    group: str
    subject: str | None
    partition: str
    label: str
    timestamps: tuple[float, ...]
    frame_hashes: tuple[str, ...]
    media_sha256: str | None
    decoded_sha256: str | None
    status: str
    media_path: str | None = None
    cache_path: str | None = None
    cache_sha256: str | None = None
    attack_family: str | None = None

    def __post_init__(self) -> None:
        if (
            self.dataset not in DOMAINS or self.partition not in PARTITIONS
            or self.label not in {"bona_fide", "attack"} or not self.key or not self.group
        ):
            raise ValueError("invalid video identity, domain, partition or label")
        if self.dataset != "SiW-Mv2" and not self.subject:
            raise ValueError("known-subject domain requires subject")
        if any(not math.isfinite(t) for t in self.timestamps):
            raise ValueError("finite timestamps required")
        if any(b <= a for a, b in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("strictly increasing native timestamps required")
        if self.frame_hashes and len(self.frame_hashes) != len(self.timestamps):
            raise ValueError("timestamp/hash count mismatch")
        if bool(self.cache_path) != bool(self.cache_sha256):
            raise ValueError("cache path and digest must be supplied together")

    @property
    def interval(self) -> tuple[float, float] | None:
        if len(self.timestamps) < 2:
            return None
        gaps = sorted(b - a for a, b in zip(self.timestamps, self.timestamps[1:]))
        return self.timestamps[0], self.timestamps[-1] + gaps[len(gaps) // 2]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(path: Path) -> tuple[list[Video], dict[str, Any]]:
    payload = read_json(path)
    if payload.get("schema_version") != 1:
        raise ValueError("manifest schema_version=1 required")
    rows = []
    for supplied in payload["videos"]:
        row = dict(supplied)
        row["timestamps"] = tuple(row["timestamps"])
        row["frame_hashes"] = tuple(row["frame_hashes"])
        rows.append(Video(**row))
    validate_isolation(rows)
    return rows, payload["provenance"]


def validate_isolation(videos: list[Video]) -> None:
    if not videos or len({v.key for v in videos}) != len(videos):
        raise ValueError("nonempty unique original identities required")
    boundaries: dict[tuple[str, ...], tuple[str, str]] = {}
    for video in videos:
        keys = [("group", video.dataset, video.group)]
        if video.subject:
            keys.append(("subject", video.dataset, video.subject))
        for value in (video.media_sha256, video.decoded_sha256, video.cache_sha256):
            if value:
                keys.append(("content", value))
        for key in keys:
            boundary = (video.dataset, video.partition)
            if key in boundaries and boundaries[key] != boundary:
                raise ValueError("group/subject/exact-content crosses partition or domain")
            boundaries[key] = boundary


def select_source(videos: list[Video], target: str, partition: str) -> list[Video]:
    if target not in DOMAINS or partition not in PARTITIONS[:3]:
        raise ValueError("known target and source partition required")
    return [v for v in videos if v.dataset != target and v.partition == partition]


def window_indices(
    timestamps: tuple[float, ...], start: float, duration: float = 2.0, fps: int = 8,
) -> tuple[int, ...]:
    if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0 or fps <= 0:
        raise ValueError("finite positive sampling duration/fps required")
    count = round(duration * fps)
    if count < 1 or not math.isclose(count, duration * fps):
        raise ValueError("duration times fps must be an integer")
    if not timestamps or any(not math.isfinite(t) for t in timestamps):
        raise ValueError("finite frame timestamps required")
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("native frame timestamps must increase strictly")
    selected = []
    for i in range(count):
        timestamp = start + (i + 0.5) / fps
        j = bisect_left(timestamps, timestamp)
        candidates = [k for k in (j - 1, j) if 0 <= k < len(timestamps)]
        chosen = min(candidates, key=lambda k: (abs(timestamps[k] - timestamp), k))
        if abs(timestamps[chosen] - timestamp) > 0.5 / fps + 1e-8:
            raise ValueError("frame exceeds half-sampling-period tolerance")
        selected.append(chosen)
    if len(set(selected)) != count:
        raise ValueError("distinct native frames required; no duplication/interpolation")
    return tuple(selected)


def clip_start(video: Video, duration: float, seed: int | None = None) -> float:
    interval = video.interval
    if interval is None or interval[1] - interval[0] + 1e-8 < duration:
        raise ValueError("insufficient_duration")
    low, high = interval[0], interval[1] - duration
    return (low + high) / 2 if seed is None else random.Random(seed).uniform(low, high)


def addressed_seed(seed: int, *address: object) -> int:
    return int(object_digest([seed, *address])[:15], 16)
