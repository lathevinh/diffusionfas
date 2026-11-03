"""Acquire and test the explicitly approved MAT mirror, then pause for crop review."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionfas.io import digest, write_json
from diffusionfas.supervision import run_bounded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    for name in ("preview-root", "upstream-root", "private-root"):
        plan.add_argument(f"--{name}", type=Path, required=True)
    plan.add_argument("--reuse-acquisition-root", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--plan-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "fas":
        raise ValueError("MAT pilot requires conda env fas")
    from diffusionfas.mat_preview import BUDGET, build_plan, checked_inputs, run_preview, summarize
    from diffusionfas.adapter_pilot import device_memory_bytes
    if args.command == "plan":
        path = build_plan(args.preview_root, args.upstream_root, args.private_root, args.reuse_acquisition_root)
        print({"plan_sha256": digest(path)}, flush=True)
    elif args.worker:
        if args.plan_sha256 is None:
            raise ValueError("MAT worker requires fixed parent plan commitment")
        run_preview(args.plan, args.plan_sha256)
    else:
        sha = digest(args.plan)
        plan = checked_inputs(args.plan, sha)
        root = args.plan.parent
        if any((root / name).exists() for name in ("batch.log", "started.json", "supervision.json")):
            raise ValueError("MAT pilot already attempted; no automatic retry")
        if device_memory_bytes() >= BUDGET["max_gpu_bytes"]:
            raise MemoryError("shared GPU already exceeds approved device-wide cap")
        supervision = run_bounded(
            [sys.executable, str(Path(__file__).resolve()), "run", "--plan", str(args.plan),
             "--worker", "--plan-sha256", sha], cwd=ROOT, owned_root=root, log=root / "batch.log",
            max_seconds=BUDGET["max_seconds"], max_rss_bytes=BUDGET["max_rss_bytes"],
            max_disk_bytes=BUDGET["max_output_bytes"], max_gpu_bytes=BUDGET["max_gpu_bytes"],
            additional_output_roots=(Path(plan["upstream_root"]),),
            gpu_memory_bytes=device_memory_bytes)
        supervision["plan_sha256"] = sha
        write_json(root / "supervision.json", supervision)
        if digest(args.plan) != sha:
            raise ValueError("MAT plan changed during run")
        report = summarize(args.plan, supervision)
        write_json(root / "aggregate.json", report)
        print({"completed": report["completed"], "aggregate_sha256": digest(root / "aggregate.json")}, flush=True)
        if not report["completed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
