"""Predetermined source-bonafide windows, exact archive staging and crop receipts."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import hashlib
import importlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
from typing import Any, BinaryIO
import zipfile

import numpy as np

from .data import Video, addressed_seed, clip_start, load_manifest, window_indices
from .fas_import import MANIFEST_SHA
from .io import digest, object_digest, read_json, require_private_output, write_json
from .preprocessing import FasFaceProvider, MediaFailure, prepare_video

SOURCES = ("CASIA-FASD", "MSU-MFSD", "SiW-Mv2")


def make_plan(metadata: Path, source_manifest: Path, fas_root: Path, output: Path) -> dict[str, Any]:
    require_private_output(output)
    videos, provenance = load_manifest(metadata)
    original = read_json(source_manifest, MANIFEST_SHA)
    if provenance["source_manifest_sha256"] != MANIFEST_SHA:
        raise ValueError("imported metadata is not the accepted FAS population")
    originals = {f"{r['dataset']}/{r['video_id']}": r for r in original["rows"]}
    selected = []
    for domain in SOURCES:
        for partition in ("fitting", "validation"):
            candidates = []
            for video in videos:
                if (video.dataset, video.partition, video.label, video.status) != (
                    domain, partition, "bona_fide", "eligible_pending_tracking",
                ):
                    continue
                try:
                    center = clip_start(video, 2)
                    window_indices(video.timestamps, center)
                except ValueError:
                    continue
                candidates.append(video)
            if not candidates:
                raise ValueError("source probe lacks declared eligible bonafide partition")
            candidates.sort(key=lambda v: object_digest(["source-probe-v1", 20261103, v.key]))
            video = candidates[0]
            row = originals[video.key]
            transaction_path = Path(row["record_path"])
            transaction = read_json(transaction_path, row["record_sha256"])
            receipt_path = transaction_path.parent.parent / "media" / transaction_path.name
            receipt = read_json(receipt_path, transaction["media_record_sha256"])
            if receipt["media_sha256"] != video.media_sha256 or receipt["binary_label"] != "bona_fide":
                raise ValueError("source probe original/label binding drift")
            starts = {
                "middle": clip_start(video, 2),
                "addressed_training_window": clip_start(video, 2, addressed_seed(20261103, video.key)),
            }
            for start in starts.values():
                window_indices(video.timestamps, start)
            selected.append({
                "video": video.to_dict(), "member": receipt["media_relpath"],
                "byte_size": receipt["byte_size"], "receipt_sha256": digest(receipt_path),
                "windows": starts,
            })
    plan = {
        "schema_version": 1, "selection_seed": 20261103, "target": "OULU-NPU",
        "selection_policy": "one minimum-address-hash eligible bonafide per source/partition before detector outcomes",
        "sources": list(SOURCES), "originals": selected, "window_seconds": 2, "fps": 8,
        "metadata_sha256": digest(metadata), "source_manifest_sha256": MANIFEST_SHA,
        "preprocessing_sha256": digest(fas_root / "src/fas/preprocessing.py"),
        "msu_reader_sha256": digest(fas_root / "scripts/inventory_msu.py"),
        "scientific_training_authorized": False, "target_decode_authorized": False,
    }
    write_json(output, plan)
    return {"selected_source_originals": len(selected), "predetermined_window_attempts": len(selected) * 2,
            "plan_sha256": digest(output), "target_originals": 0}


def safe_member(member: str) -> None:
    path = PurePosixPath(member)
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", member) or path.is_absolute() or ".." in path.parts or member.startswith("-"):
        raise ValueError("unsafe archive member")


def copy_member(source: BinaryIO, output: Path, expected_bytes: int, expected_sha: str, deadline: float) -> None:
    if expected_bytes <= 0 or expected_bytes > 128 * 1024**2:
        raise ValueError("probe member byte cap")
    checksum, count = hashlib.sha256(), 0
    with output.open("xb") as target:
        while chunk := source.read(1024 * 1024):
            count += len(chunk)
            if count > expected_bytes or time.monotonic() > deadline:
                raise ValueError("source archive byte/deadline guard")
            checksum.update(chunk)
            target.write(chunk)
    if count != expected_bytes or checksum.hexdigest() != expected_sha:
        raise ValueError("staged source video differs from accepted bytes")


def stage_member(
    item: dict[str, Any], fas_root: Path, data_root: Path, output: Path,
    stack: ExitStack, msu_context: dict[str, Any], deadline: float,
) -> None:
    video, member = item["video"], item["member"]
    if video["dataset"] not in SOURCES or video["label"] != "bona_fide" or video["partition"] not in {"fitting", "validation"}:
        raise ValueError("staging restricted to predetermined source bonafide")
    safe_member(member)
    expected_size, expected_sha = item["byte_size"], video["media_sha256"]
    if video["dataset"] == "SiW-Mv2":
        with zipfile.ZipFile(data_root / "SiW/SiW-Mv2.zip") as archive:
            entry = archive.getinfo(member)
            if entry.is_dir() or entry.file_size != expected_size or entry.flag_bits & 1:
                raise ValueError("unexpected SiW source member")
            with archive.open(entry) as source:
                copy_member(source, output, expected_size, expected_sha, deadline)
    elif video["dataset"] == "CASIA-FASD":
        extractor = Path(sys.prefix).parent.parent / "bin/bsdtar"
        if not extractor.is_file():
            raise ValueError("existing bsdtar extractor unavailable; no installation fallback")
        errors = output.with_suffix(output.suffix + ".stderr")
        with errors.open("xb") as log:
            process = subprocess.Popen(
                [str(extractor), "-xOf", str(data_root / "casia-fasd/CASIA_faceAntisp.rar"), "--", member],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log,
            )
            try:
                if process.stdout is None:
                    raise ValueError("archive extractor stdout unavailable")
                copy_member(process.stdout, output, expected_size, expected_sha, deadline)
                code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
                if code or errors.stat().st_size:
                    raise ValueError("strict RAR member extraction reported failure")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if process.stdout is not None:
                    process.stdout.close()
    else:
        if "archive" not in msu_context:
            sys.path.insert(0, str(fas_root / "scripts"))
            native = importlib.import_module("inventory_msu")
            if Path(native.__file__).resolve() != (fas_root / "scripts/inventory_msu.py").resolve():
                raise ValueError("MSU reader imported from an unbound location")
            parts = {}
            for archive_path in sorted((data_root / "MSU").glob("*.zip")):
                with zipfile.ZipFile(archive_path) as outer:
                    for entry in outer.infolist():
                        native.safe_entry(entry)
                        match = re.fullmatch(r"MSU-MFSD/MSU-MFSD-Publish\.zip\.([0-9]{3})", entry.filename)
                        if match:
                            number = int(match.group(1))
                            if number in parts:
                                raise ValueError("duplicate MSU volume")
                            parts[number] = (archive_path, entry)
            if set(parts) != set(range(1, 17)):
                raise ValueError("exact sixteen MSU native split volumes required")
            reader = native.SplitZipReader([parts[i] for i in range(1, 17)], stack)
            msu_context["archive"] = stack.enter_context(zipfile.ZipFile(reader))
        archive = msu_context["archive"]
        entry = archive.getinfo(member)
        if entry.is_dir() or entry.file_size != expected_size or entry.flag_bits & 1:
            raise ValueError("unexpected MSU source member")
        with archive.open(entry) as source:
            copy_member(source, output, expected_size, expected_sha, deadline)


def run_source_probe(
    plan_path: Path, fas_root: Path, data_root: Path, detector: Path, output: Path,
    *, max_seconds: float = 900, previous_probe: Path | None = None,
    source_manifest: Path | None = None,
) -> dict[str, Any]:
    require_private_output(output / "placeholder")
    plan = read_json(plan_path)
    if output.exists():
        raise ValueError("new source probe output directory required")
    if digest(fas_root / "scripts/inventory_msu.py") != plan["msu_reader_sha256"]:
        raise ValueError("source archive reader changed since selection")
    if len(plan["originals"]) != 6 or plan["target"] != "OULU-NPU":
        raise ValueError("exact fixed six-source-original probe required")
    previous, native, native_hash, original_rows = None, None, None, {}
    if previous_probe is not None:
        previous = read_json(previous_probe)
        if previous["plan_sha256"] != digest(plan_path) or source_manifest is None:
            raise ValueError("native replay must preserve the initial committed selection")
        failures = [a for a in previous["attempts"] if not a["success"]]
        if len(failures) != 4 or any(a["dataset"] != "MSU-MFSD" or a["failure"] !=
                                  "native_RGB_hash_mismatch_no_decoder_waiver" for a in failures):
            raise ValueError("native follow-up is restricted to the four observed MSU RGB failures")
        original_rows = {f"{r['dataset']}/{r['video_id']}": r
                         for r in read_json(source_manifest, MANIFEST_SHA)["rows"]}
        sys.path.insert(0, str(fas_root / "scripts"))
        sys.path.insert(0, str(fas_root / "src"))
        native = importlib.import_module("msu_replay282")
        native_path = fas_root / "scripts/msu_replay282.py"
        if Path(native.__file__).resolve() != native_path.resolve():
            raise ValueError("unbound source native decoder")
        native_hash = digest(native_path)
    output.mkdir(parents=True)
    temporary = output / "staging"
    temporary.mkdir()
    provider = FasFaceProvider(
        fas_root, plan["preprocessing_sha256"], detector,
        "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    )
    started, deadline, attempts = time.monotonic(), time.monotonic() + max_seconds, []
    with ExitStack() as stack:
        msu_context: dict[str, Any] = {}
        for ordinal, item in enumerate(plan["originals"]):
            if previous is not None and item["video"]["dataset"] != "MSU-MFSD":
                continue
            values = dict(item["video"])
            values["timestamps"], values["frame_hashes"] = tuple(values["timestamps"]), tuple(values["frame_hashes"])
            video = Video(**values)
            staged = temporary / f"source-{ordinal}{Path(item['member']).suffix}"
            before = time.monotonic()
            stage_member(item, fas_root, data_root, staged, stack, msu_context, deadline)
            stage_seconds = time.monotonic() - before
            bound = replace(video, media_path=str(staged))
            try:
                for window_name, start in item["windows"].items():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("source probe deadline")
                    crop_file = output / f"crop-{ordinal}-{window_name}.npz"
                    entry: dict[str, Any] = {
                        "source_ordinal": ordinal, "dataset": video.dataset, "partition": video.partition,
                        "window": window_name, "stage_seconds": stage_seconds,
                        "source_bytes_verified": True,
                    }
                    window_started = time.monotonic()
                    replay = None
                    if native is not None:
                        transaction_path = Path(original_rows[video.key]["record_path"])
                        transaction = read_json(transaction_path, original_rows[video.key]["record_sha256"])
                        media_path = transaction_path.parent.parent / "media" / transaction_path.name
                        receipt = read_json(media_path, item["receipt_sha256"])
                        if transaction["media_record_sha256"] != item["receipt_sha256"]:
                            raise ValueError("source replay receipt parent drift")
                        replays = [0]

                        def replay(path, consume, timeout):
                            pass_number = replays[0]
                            replays[0] += 1
                            try:
                                native.strict_replay(
                                    path, receipt, consume,
                                    output / f"native-{ordinal}-{window_name}-pass-{pass_number}.json",
                                    timeout=timeout,
                                )
                            except TimeoutError as error:
                                raise MediaFailure("native_ffmpeg_deadline") from error
                            except ValueError as error:
                                if str(error) not in {"RGB identity/count mismatch", "extra RGB bytes"} and not str(error).startswith("strict decoder returncode="):
                                    raise
                                raise MediaFailure("native_ffmpeg_replay_failure") from error
                    try:
                        result = prepare_video(bound, provider, crop_file, max_seconds=min(
                            120, deadline - time.monotonic()), window_start=start, window_duration=2,
                            native_replay=replay)
                    except MediaFailure as error:
                        entry.update(success=False, failure=str(error), crop_sha256=None)
                    else:
                        with np.load(result.cache_path, allow_pickle=False) as cache:
                            if cache["frames"].shape != (16, 256, 256, 3):
                                raise ValueError("probe did not produce the required 2s/16-frame crop shape")
                        entry.update(success=True, failure=None, crop_sha256=result.cache_sha256)
                    entry["wall_seconds"] = time.monotonic() - window_started
                    attempts.append(entry)
                    write_json(output / f"attempt-{ordinal}-{window_name}.json", entry)
            finally:
                staged.unlink(missing_ok=True)
    report = {
        "schema_version": 1, "completed": True, "plan_sha256": digest(plan_path),
        "source_originals": len({a["source_ordinal"] for a in attempts}),
        "window_attempts": len(attempts), "attempts": attempts,
        "successful_windows": sum(a["success"] for a in attempts),
        "wall_seconds": time.monotonic() - started, "target_video_decodes": 0,
        "scientific_training_runs": 0, "source_model_scores": 0, "target_evaluations": 0,
        "selection_replacements": 0, "native_RGB_waivers": 0,
        "scope": "engineering_preprocessing_probe_not_PAD_accuracy",
        "msu_split_container_seek_may_decompress_other_container_bytes": True,
        "native_replay_helper_sha256": native_hash,
        "previous_probe_sha256": digest(previous_probe) if previous_probe is not None else None,
        "decoder": "accepted_FFmpeg_strict_full_RGB_replay" if native is not None else "OpenCV",
    }
    write_json(output / "source-probe.json", report)
    return report
