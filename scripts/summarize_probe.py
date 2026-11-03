"""Publish aggregate probe evidence without original IDs, paths, crops or PAD scores."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionfas.data import window_indices
from diffusionfas.io import digest, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.private_root
    names = {
        "acquisition": "pretrained/acquisition.json", "gpu": "gpu/gpu-probe.json",
        "source_initial": "source/source-probe.json",
        "source_native_followup": "source-native/source-probe.json", "replay": "replay.json",
        "native_microbatch": "microbatch.json",
    }
    receipts = {key: read_json(root / name) for key, name in names.items()}
    if not all(value["completed"] for value in receipts.values()):
        raise ValueError("all requested probes must have completed before aggregate publication")
    plan = read_json(root / "source-plan.json")
    initial, native = receipts["source_initial"], receipts["source_native_followup"]
    if (
        initial["plan_sha256"] != digest(root / "source-plan.json")
        or native["plan_sha256"] != initial["plan_sha256"]
        or native["previous_probe_sha256"] != digest(root / names["source_initial"])
        or receipts["gpu"]["bundle_manifest_sha256"] != receipts["acquisition"]["bundle_manifest_sha256"]
        or len(initial["attempts"]) != 12 or len(native["attempts"]) != 4
    ):
        raise ValueError("probe lineage or exact attempted-window population differs")
    final = {(a["source_ordinal"], a["window"]): a for a in initial["attempts"]}
    for attempt in native["attempts"]:
        key = (attempt["source_ordinal"], attempt["window"])
        if key not in final or final[key]["success"] or final[key]["dataset"] != "MSU-MFSD":
            raise ValueError("native replay changed the predetermined failing-window population")
        final[key] = attempt
    if len(final) != 12 or not all(a["success"] for a in final.values()):
        raise ValueError("not all twelve predetermined windows passed strict replay/tracking")
    native_checks = [read_json(path) for path in sorted((root / "source-native").glob("native-*-pass-*.json"))]
    if len(native_checks) != 8 or not all(r["strict_success"] and r["frames_verified"] == r["frame_count"]
                                          for r in native_checks):
        raise ValueError("native follow-up must verify both full decode passes for every MSU window")
    supervision = {path.stem.removeprefix("supervision-"): read_json(path)
                   for path in sorted(root.glob("supervision-*.json"))}
    native_completions = [name for name, result in supervision.items()
                          if name.startswith("source-native") and result["completed"]]
    if len(native_completions) != 1:
        raise ValueError("exactly one successful explicitly bounded native follow-up required")
    for phase in ("plan", "acquire", "gpu", "source", "replay", "microbatch"):
        if phase not in supervision or not supervision[phase]["completed"]:
            raise ValueError("required bounded phase did not complete successfully")
    duplicates = 0
    for item in plan["originals"]:
        timestamps = tuple(item["video"]["timestamps"])
        starts = list(item["windows"].values())
        duplicates += window_indices(timestamps, starts[0]) == window_indices(timestamps, starts[1])
    by_domain = {}
    for domain in plan["sources"]:
        first = [a for a in initial["attempts"] if a["dataset"] == domain]
        followup = [a for a in native["attempts"] if a["dataset"] == domain]
        by_domain[domain] = {
            "selected_originals": 2, "fitting_originals": 1, "validation_originals": 1,
            "fixed_unique_window_slots": 4, "initial_attempts": len(first),
            "initial_successful_windows": sum(a["success"] for a in first),
            "initial_failures": dict(Counter(a["failure"] for a in first if not a["success"])),
            "explicit_native_followup_attempts": len(followup),
            "final_successful_windows": sum(a["dataset"] == domain for a in final.values()),
        }
    config = read_json(ROOT / "configs/offline-v1.json")
    if any(config["execution"].values()):
        raise ValueError("scientific execution gates must remain disabled")
    report = {
        "schema_version": 1, "completed": True, "environment": "conda:fas",
        "scope": "engineering_feasibility_only_not_PAD_accuracy_or_cross_domain_generalization",
        "acquisition": receipts["acquisition"], "synthetic_gpu": receipts["gpu"],
        "separate_process_replay": receipts["replay"],
        "native_pretrained_microbatch_equivalence": receipts["native_microbatch"],
        "source_preprocessing": {
            "selection_plan_sha256": initial["plan_sha256"], "selection_seed": plan["selection_seed"],
            "selected_originals": 6, "predetermined_window_slots": 12,
            "pairs_with_identical_native_frame_indices": duplicates,
            "window_seconds": 2, "frames_per_window": 16, "fps": 8, "RGB_shape": [16, 256, 256, 3],
            "initial_window_attempts": 12, "initial_successful_windows": initial["successful_windows"],
            "explicit_native_followup_attempts": 4, "total_window_attempts_including_repair": 16,
            "final_successful_window_slots": 12, "by_domain": by_domain,
            "initial_decoder": "OpenCV", "MSU_followup_decoder": native["decoder"],
            "native_replay_helper_sha256": native["native_replay_helper_sha256"],
            "MSU_strict_full_decode_passes": len(native_checks),
            "MSU_RGB_frames_verified_across_all_followup_passes": sum(r["frames_verified"] for r in native_checks),
            "initial_preprocessing_wall_seconds": initial["wall_seconds"],
            "native_followup_wall_seconds": native["wall_seconds"],
            "original_or_window_replacements": 0, "native_RGB_waivers": 0,
            "source_model_forwards": 0, "target_video_decodes": 0,
            "MSU_seeks_may_decompress_other_container_bytes": True,
            "newly_staged_originals_removed": not any(
                path.suffix.lower() in {".avi", ".mp4", ".mov"}
                for directory in ("source/staging", "source-native/staging")
                for path in (root / directory).iterdir()
            ),
        },
        "supervision": supervision,
        "failed_supervised_invocations_preserved": sum(not r["completed"] for r in supervision.values()),
        "failed_invocation_explanation": (
            "first native-followup worker failed before staging/decoding because the reused FAS package was not on sys.path; corrected explicit attempt retained both immutable receipts"
            if "source-native" in supervision and not supervision["source-native"]["completed"] else None
        ),
        "guard_semantics": "0.2s sampled process-tree RSS/owned-disk watchdog, finite wall limit with up-to-5s termination grace; not OS-enforced instantaneous RSS/disk quotas; CUDA limit covers own PyTorch allocator, not other users or all driver allocations",
        "private_receipt_sha256": {key: digest(root / name) for key, name in names.items()},
        "scientific_training_runs": 0, "PAD_target_evaluations": 0,
        "full_source_preprocessing_runs": 0, "scientific_execution_flags": config["execution"],
        "calibration_policy_changed": False,
        "calibration_blocker": "CASIA 12 and MSU 6 bonafide calibration originals cannot support the chosen strict 5% policy requiring at least one rejection unit; splits/rate unchanged",
        "limitations": [
            "six originals and twelve unique slots cannot establish dataset-wide tracking success or PAD discrimination",
            "three measured synthetic steps per arm do not establish end-to-end training runtime",
            "CUDA training bitwise determinism is not claimed",
            "OpenCV CLI default remains unchanged; accepted native replay is explicitly injectable into prepare_video and used for the fixed MSU follow-up",
            "no DINO or scratch-model performance benchmark was executed",
            "model-card license review is not unrestricted data/commercial-rights certification",
        ],
    }
    if not report["source_preprocessing"]["newly_staged_originals_removed"]:
        raise ValueError("newly staged source originals were not cleaned up")
    write_json(arguments.out, report)
    print({"completed": True, "final_unique_window_slots": len(final),
           "aggregate_receipt_sha256": digest(arguments.out)})


if __name__ == "__main__":
    main()
