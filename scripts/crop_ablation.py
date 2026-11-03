"""Run one explicitly approved, bounded frozen-DDPM MSU crop comparison."""

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
    plan.add_argument("--preview-root", type=Path, required=True)
    plan.add_argument("--private-root", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--plan-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "fas":
        raise ValueError("crop ablation requires conda env fas")
    from diffusionfas.crop_ablation import BUDGET, build_plan, checked_inputs, run_ablation, summarize
    from diffusionfas.adapter_pilot import device_memory_bytes
    if args.command == "plan":
        path = build_plan(args.preview_root, args.private_root)
        print({"plan_sha256": digest(path)}, flush=True)
    elif args.worker:
        if args.plan_sha256 is None:
            raise ValueError("fixed parent plan commitment required")
        run_ablation(args.plan, args.plan_sha256)
    else:
        sha = digest(args.plan)
        checked_inputs(args.plan, sha)
        root = args.plan.parent
        if any((root / name).exists() for name in ("batch.log", "supervision.json", "started.json")):
            raise ValueError("ablation already attempted; no automatic retry")
        if device_memory_bytes() >= BUDGET["max_gpu_bytes"]:
            raise MemoryError("shared device already exceeds conservative GPU cap")
        supervision = run_bounded(
            [sys.executable, str(Path(__file__).resolve()), "run", "--plan", str(args.plan),
             "--worker", "--plan-sha256", sha], cwd=ROOT, owned_root=root, log=root / "batch.log",
            max_seconds=BUDGET["max_seconds"], max_rss_bytes=BUDGET["max_rss_bytes"],
            max_disk_bytes=BUDGET["max_output_bytes"], max_gpu_bytes=BUDGET["max_gpu_bytes"],
            gpu_memory_bytes=device_memory_bytes)
        supervision["plan_sha256"] = sha
        write_json(root / "supervision.json", supervision)
        if digest(args.plan) != sha:
            raise ValueError("ablation plan changed during run")
        report = summarize(args.plan, supervision)
        write_json(root / "aggregate.json", report)
        print({"completed": report["completed"], "aggregate_sha256": digest(root / "aggregate.json")}, flush=True)
        if not report["completed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
