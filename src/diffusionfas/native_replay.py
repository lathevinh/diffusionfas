"""Receipt-bound reuse of the accepted FAS strict FFmpeg decoder."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .data import DOMAINS, Video
from .fas_import import read_media_receipt
from .io import digest, read_json, require_private_output
from .preprocessing import MediaFailure, NativeReplay

FAS_NATIVE_HELPER_SHA256 = "8018390bfa8004d42cd52eae21dab8633a39838f1efc523647de5cd9d823a888"


def decoder_policy(config: dict[str, Any]) -> tuple[dict[str, str], str | None]:
    settings = config.get("preprocessing", {})
    if not isinstance(settings, dict) or set(settings) - {"decoders", "native_replay_helper_sha256"}:
        raise ValueError("preprocessing must be a decoder-policy object")
    configured = settings.get("decoders", {})
    if not isinstance(configured, dict) or set(configured) - set(DOMAINS):
        raise ValueError("decoder policy must name only known domains")
    if any(value not in ("opencv", "fas_strict_ffmpeg") for value in configured.values()):
        raise ValueError("unknown preprocessing decoder; no automatic fallback")
    decoders = {domain: configured.get(domain, "opencv") for domain in DOMAINS}
    if decoders["OULU-NPU"] == "fas_strict_ffmpeg":
        raise ValueError("native receipt bridge currently supports non-OULU domains only")
    helper_sha = settings.get("native_replay_helper_sha256")
    if ("fas_strict_ffmpeg" in decoders.values() or helper_sha is not None) and (
        not isinstance(helper_sha, str) or len(helper_sha) != 64
        or any(c not in "0123456789abcdef" for c in helper_sha)
    ):
        raise ValueError("strict FFmpeg decoder requires an explicit helper SHA-256")
    return decoders, helper_sha


class FasNativeReplay:
    def __init__(
        self, source_root: Path, helper_sha256: str,
        source_manifest: Path, source_manifest_sha256: str,
    ) -> None:
        helper = source_root / "scripts/msu_replay282.py"
        if digest(helper) != helper_sha256:
            raise ValueError("native replay helper hash mismatch")
        manifest = read_json(source_manifest, source_manifest_sha256)
        self.rows = {f"{r['dataset']}/{r['video_id']}": r for r in manifest["rows"]}
        if len(self.rows) != len(manifest["rows"]):
            raise ValueError("duplicate native source original")
        sys.path.insert(0, str(source_root / "scripts"))
        sys.path.insert(0, str(source_root / "src"))
        self.module = importlib.import_module("msu_replay282")
        if Path(self.module.__file__).resolve() != helper.resolve():
            raise ValueError("native decoder imported from an unbound location")
        self.helper_sha256 = helper_sha256
        self.source_manifest_sha256 = source_manifest_sha256

    def bind(
        self, video: Video, prefix: Path, *, expected_receipt_sha256: str | None = None,
    ) -> NativeReplay:
        require_private_output(prefix.with_suffix(".json"))
        if video.dataset == "OULU-NPU" or video.key not in self.rows:
            raise ValueError("native replay requires a known non-OULU original")
        row = self.rows[video.key]
        if row["binary_label"] != video.label:
            raise ValueError("native replay original label mismatch")
        receipt, receipt_sha = read_media_receipt(row)
        if expected_receipt_sha256 is not None and receipt_sha != expected_receipt_sha256:
            raise ValueError("native replay receipt differs from committed selection")
        frames = receipt["frame_index"]
        if (
            receipt["decode_status"] != "decoded" or receipt["frame_count"] != len(video.timestamps)
            or receipt["media_sha256"] != video.media_sha256
            or receipt["decoded_rgb_sequence_sha256"] != video.decoded_sha256
            or tuple(f["rgb_sha256"] for f in frames) != video.frame_hashes
            or tuple(float(Fraction(f["timestamp_seconds"])) for f in frames) != video.timestamps
        ):
            raise ValueError("native replay receipt differs from imported video evidence")
        tokens = self.module.decoded_tokens(receipt)
        if len({token[:2] for token in tokens}) != 1:
            raise ValueError("native replay requires constant accepted frame dimensions")
        passes = 0

        def replay(path: Path, consume: Callable[[int, np.ndarray], None], timeout: float) -> None:
            nonlocal passes
            if digest(path) != video.media_sha256:
                raise ValueError("native replay media bytes differ from bound original")
            receipt_output = prefix.parent / f"{prefix.name}-pass-{passes}.json"
            passes += 1
            if receipt_output.exists() or receipt_output.with_suffix(".stderr").exists():
                raise ValueError("new native replay receipt paths required")
            try:
                self.module.strict_replay(path, receipt, consume, receipt_output, timeout=timeout)
            except TimeoutError as error:
                raise MediaFailure("native_ffmpeg_deadline") from error
            except ValueError as error:
                if (
                    str(error) not in {"RGB identity/count mismatch", "extra RGB bytes", "decoder stderr16MiB cap"}
                    and not str(error).startswith("strict decoder returncode=")
                ):
                    raise
                raise MediaFailure("native_ffmpeg_replay_failure") from error

        return replay
