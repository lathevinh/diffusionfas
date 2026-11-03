"""Explicit bounded research probes; no real-data model training or target access."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionfas.io import digest, read_json, require_private_output, write_json
from diffusionfas.supervision import run_bounded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "acquire", "gpu", "source", "source-native", "replay", "microbatch"))
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--fas-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--attempt-label", default="initial")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "fas":
        raise ValueError("all probes must run in conda env fas")
    require_private_output(arguments.private_root / "placeholder")
    arguments.private_root.mkdir(parents=True, exist_ok=True)
    phase = arguments.phase
    if not re.fullmatch(r"[a-z0-9-]+", arguments.attempt_label):
        raise ValueError("safe explicit probe attempt label required")
    suffix = "" if arguments.attempt_label == "initial" else f"-{arguments.attempt_label}"
    limits = {"plan": 180, "acquire": 900, "gpu": 600, "source": 1200, "source-native": 600,
              "replay": 300, "microbatch": 120}
    if not arguments.worker:
        receipt = arguments.private_root / f"supervision-{phase}{suffix}.json"
        if receipt.exists():
            raise ValueError("phase already attempted; no automatic retry or receipt overwrite")
        result = run_bounded(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
            cwd=ROOT, owned_root=arguments.private_root,
            log=arguments.private_root / f"{phase}{suffix}.log",
            max_seconds=limits[phase], max_rss_bytes=12 * 1024**3, max_disk_bytes=8 * 1024**3,
        )
        result["limits"] = {"wall_seconds": limits[phase], "rss_bytes": 12 * 1024**3,
                            "owned_disk_bytes": 8 * 1024**3}
        write_json(receipt, result)
        print(result, flush=True)
        if not result["completed"]:
            raise SystemExit(1)
        return
    if phase == "plan":
        from diffusionfas.source_probe import make_plan

        make_plan(arguments.metadata, arguments.source_manifest, arguments.fas_root,
                  arguments.private_root / "source-plan.json")
    elif phase == "acquire":
        from diffusionfas.acquisition import acquire_bundle

        acquire_bundle(arguments.private_root / "pretrained", max_seconds=limits[phase] - 15)
    elif phase in {"source", "source-native"}:
        from diffusionfas.source_probe import run_source_probe

        run_source_probe(arguments.private_root / "source-plan.json", arguments.fas_root,
                         arguments.fas_root, arguments.detector, arguments.private_root / phase,
                         max_seconds=limits[phase] - 15,
                         previous_probe=arguments.private_root / "source/source-probe.json"
                         if phase == "source-native" else None,
                         source_manifest=arguments.source_manifest)
    else:
        from diffusionfas.gpu_probe import replay_gpu_probe, run_gpu_probe, verify_native_microbatch

        acquisition = read_json(arguments.private_root / "pretrained/acquisition.json")
        bundle = arguments.private_root / "pretrained/bundle"
        if digest(bundle / "manifest.json") != acquisition["bundle_manifest_sha256"]:
            raise ValueError("acquired bundle receipt mismatch")
        if phase == "gpu":
            run_gpu_probe(bundle, acquisition["bundle_manifest_sha256"], arguments.private_root / "gpu")
        elif phase == "microbatch":
            verify_native_microbatch(bundle, acquisition["bundle_manifest_sha256"],
                                     arguments.private_root / "gpu", arguments.private_root / "microbatch.json")
        else:
            replay_gpu_probe(bundle, acquisition["bundle_manifest_sha256"],
                             arguments.private_root / "gpu", arguments.private_root / "replay.json")


if __name__ == "__main__":
    main()
