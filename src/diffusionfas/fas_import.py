"""Read accepted FAS metadata; never decode media or reallocate partitions."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from .data import Video, validate_isolation
from .io import digest, read_json

MANIFEST_SHA = "7f9f54e6b097d7da0618cf8b3784e9eb3726b6aba8a3957455e26dda20da4580"
PARTITIONS_SHA = "2700baf9d757cc8c773247f104cd2a85c4ee9c288e461bb929cdbff9333c2996"
SOURCE_REVISION = "707e3ddf6103835c55e937cbbd84081d91f188c7"


def read_media_receipt(
    row: dict[str, Any], oulu_media_root: Path | None = None,
) -> tuple[dict[str, Any], str]:
    transaction_path = Path(row["record_path"])
    transaction = read_json(transaction_path, row["record_sha256"])
    if transaction["video_id"] != row["video_id"] or transaction["binary_label"] != row["binary_label"]:
        raise ValueError("transaction original/label mismatch")
    if row["dataset"] == "OULU-NPU":
        if oulu_media_root is None:
            raise ValueError("OULU media receipt root required")
        receipt_path = oulu_media_root / f"{row['video_id']}.json"
    else:
        receipt_path = transaction_path.parent.parent / "media" / transaction_path.name
    receipt = read_json(receipt_path, transaction["media_record_sha256"])
    if receipt["media_sha256"] != transaction["media_sha256"]:
        raise ValueError("media receipt/transaction binding mismatch")
    if receipt["video_id"] != row["video_id"]:
        raise ValueError("media receipt identity mismatch")
    return receipt, transaction["media_record_sha256"]


def import_metadata(
    manifest_path: Path, partitions_path: Path, oulu_media_root: Path,
    manifest_sha: str = MANIFEST_SHA, partitions_sha: str = PARTITIONS_SHA,
) -> tuple[list[Video], dict[str, Any]]:
    manifest = read_json(manifest_path, manifest_sha)
    partitions = read_json(partitions_path, partitions_sha)
    if partitions["source_manifest_sha256"] != manifest_sha:
        raise ValueError("partition/source manifest binding mismatch")
    contexts = partitions["contexts"]
    roles: dict[str, str] = {}
    for context in contexts:
        for role in ("fitting", "validation", "calibration", "evaluation"):
            for key in context[role]:
                if key in roles and roles[key] != role:
                    raise ValueError("original changes partition across contexts")
                roles[key] = role
    rows, excluded, receipt_counts = [], 0, Counter()
    for row in manifest["rows"]:
        if row["dataset"] == "SiW-Mv2" and row["official_split"] == "train" and not row["source_use_eligible"]:
            excluded += 1
            continue
        key = f"{row['dataset']}/{row['video_id']}"
        if key not in roles:
            raise ValueError("active original missing from accepted partitions")
        if (roles[key] == "evaluation") != (row["official_split"] == "test"):
            raise ValueError("official test/experiment partition mismatch")
        if row["record_path"] is None:
            if row["record_sha256"] is not None or row["detector_status"] != "not_run" or row["failure_kind"] != "pre_detector":
                raise ValueError("missing transaction without accepted pre-detector failure evidence")
            rows.append(Video(
                key=key, dataset=row["dataset"], group=str(row["group_id"]),
                subject=row["subject_id"] or None, partition=roles[key], label=row["binary_label"],
                timestamps=(), frame_hashes=(), media_sha256=None, decoded_sha256=None,
                status="historical_pre_detector_failure",
            ))
            continue
        receipt, _ = read_media_receipt(row, oulu_media_root)
        receipt_counts[row["dataset"]] += 1
        frames = receipt.get("frame_index", [])
        status = "eligible_pending_tracking" if receipt["decode_status"] == "decoded" else "historical_decode_failure"
        timestamps = tuple(float(Fraction(frame["timestamp_seconds"])) for frame in frames)
        hashes = tuple(frame["rgb_sha256"] for frame in frames)
        rows.append(Video(
            key=key, dataset=row["dataset"], group=str(row["group_id"]),
            subject=row["subject_id"] or None, partition=roles[key], label=row["binary_label"],
            timestamps=timestamps, frame_hashes=hashes, media_sha256=receipt["media_sha256"],
            decoded_sha256=receipt.get("decoded_rgb_sequence_sha256"), status=status,
        ))
    if len(rows) != len(roles):
        raise ValueError("accepted role population and imported originals differ")
    validate_isolation(rows)
    return rows, {
        "source_repository": "lathevinh/fas", "source_revision": SOURCE_REVISION,
        "source_manifest_sha256": digest(manifest_path),
        "partitions_sha256": digest(partitions_path),
        "source_use_binding": manifest["source_use_binding"],
        "excluded_source_originals": excluded, "verified_media_receipts": dict(receipt_counts),
        "new_video_decodes": 0, "new_detector_calls": 0, "model_forwards": 0,
        "temporal_tracking_verified": False, "extracted_media_paths_bound": False,
        "interval_policy": "accepted_native_receipt_frame_index_no_new_temporal_bounds",
    }
