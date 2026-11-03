"""Explicit 45-minute matched source-only adapter pilot in conda fas."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionfas.io import digest, read_json, write_json
from diffusionfas.supervision import run_bounded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    for name in ("manifest", "preparation-plan", "bundle", "private-root"):
        plan.add_argument(f"--{name}", type=Path, required=True)
    for name in ("manifest-sha256", "bundle-sha256"):
        plan.add_argument(f"--{name}", required=True)
    plan.add_argument("--config", type=Path, default=ROOT / "configs/offline-v1.json")
    for name in ("run", "replay", "report"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        if name != "report":
            command.add_argument("--plan-sha256")
            command.add_argument("--training-result-sha256")
            command.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
        else:
            command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "fas":
        raise ValueError("adapter pilot requires conda env fas")
    from diffusionfas.adapter_pilot import build_plan, device_memory_bytes, replay_pilot, run_pilot, summarize

    if args.command == "plan":
        path = build_plan(args.manifest, args.manifest_sha256, args.preparation_plan, args.bundle,
                          args.bundle_sha256, args.config, args.private_root)
        print({"plan_sha256": digest(path)}, flush=True)
    elif args.command == "report":
        supervision = read_json(args.plan.parent / "supervision.json")
        if supervision["plan_sha256"] != digest(args.plan):
            raise ValueError("supervised pilot commitment changed")
        report = summarize(args.plan, supervision)
        write_json(args.out, report)
        print({"completed": report["completed"], "aggregate_sha256": digest(args.out)}, flush=True)
    elif args.command == "replay":
        if args.plan_sha256 is None or args.training_result_sha256 is None:
            raise ValueError("replay requires fixed plan/result commitments")
        replay_pilot(args.plan, args.plan_sha256, args.training_result_sha256)
    elif args.worker:
        if args.plan_sha256 is None:
            raise ValueError("worker requires fixed plan commitment")
        run_pilot(args.plan, args.plan_sha256, Path(__file__).resolve())
    else:
        sha = digest(args.plan)
        plan = read_json(args.plan, sha)
        root = args.plan.parent
        if (root / "batch.log").exists() or (root / "supervision.json").exists():
            raise ValueError("pilot already attempted; no automatic retry")
        budget = plan["budget"]
        if device_memory_bytes() >= budget["max_gpu_bytes"]:
            raise MemoryError("shared device already exceeds approved conservative GPU cap")
        result = run_bounded(
            [sys.executable, str(Path(__file__).resolve()), "run", "--plan", str(args.plan),
             "--worker", "--plan-sha256", sha],
            cwd=ROOT, owned_root=root, log=root / "batch.log", max_seconds=budget["max_seconds"],
            max_rss_bytes=budget["max_rss_bytes"], max_disk_bytes=budget["max_output_bytes"],
            max_gpu_bytes=budget["max_gpu_bytes"], gpu_memory_bytes=device_memory_bytes,
        )
        result["plan_sha256"] = sha
        write_json(root / "supervision.json", result)
        if digest(args.plan) != sha:
            raise ValueError("pilot plan changed during execution")
        report = summarize(args.plan, result)
        write_json(root / "aggregate.json", report)
        print({"completed": report["completed"], "aggregate_sha256": digest(root / "aggregate.json")},
              flush=True)
        if not report["completed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
