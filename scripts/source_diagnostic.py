"""Explicit source-validation development diagnostic with fixed source-trained models."""

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
    for name in ("manifest", "pilot-root", "source-manifest", "fas-root", "detector", "private-root"):
        plan.add_argument(f"--{name}", type=Path, required=True)
    for name in ("run", "report"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        if name == "run":
            command.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
            command.add_argument("--plan-sha256", help=argparse.SUPPRESS)
        else:
            command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "fas":
        raise ValueError("source diagnostic requires conda env fas")
    from diffusionfas.adapter_pilot import device_memory_bytes
    from diffusionfas.source_diagnostic import BUDGET, build_plan, run_diagnostic, summarize

    if args.command == "plan":
        path = build_plan(args.manifest, args.pilot_root, args.source_manifest, args.fas_root,
                          args.detector, args.private_root)
        print({"plan_sha256": digest(path)}, flush=True)
    elif args.command == "report":
        supervision = read_json(args.plan.parent / "supervision.json")
        if supervision["plan_sha256"] != digest(args.plan):
            raise ValueError("supervised diagnostic plan commitment changed")
        report = summarize(args.plan, supervision)
        write_json(args.out, report)
        print({"completed": report["completed"], "aggregate_sha256": digest(args.out)}, flush=True)
    elif args.worker:
        if args.plan_sha256 is None:
            raise ValueError("worker requires fixed parent plan commitment")
        run_diagnostic(args.plan, args.plan_sha256)
    else:
        sha = digest(args.plan)
        plan = read_json(args.plan, sha)
        root = args.plan.parent
        if plan["budget"] != BUDGET:
            raise ValueError("diagnostic caps differ from explicit approval")
        if (root / "batch.log").exists() or (root / "supervision.json").exists():
            raise ValueError("diagnostic already attempted; no automatic retry")
        if device_memory_bytes() >= BUDGET["max_gpu_bytes"]:
            raise MemoryError("shared device already exceeds conservative GPU cap")
        result = run_bounded(
            [sys.executable, str(Path(__file__).resolve()), "run", "--plan", str(args.plan),
             "--worker", "--plan-sha256", sha],
            cwd=ROOT, owned_root=root, log=root / "batch.log", max_seconds=BUDGET["max_seconds"],
            max_rss_bytes=BUDGET["max_rss_bytes"], max_disk_bytes=BUDGET["max_output_bytes"],
            max_gpu_bytes=BUDGET["max_gpu_bytes"], gpu_memory_bytes=device_memory_bytes,
        )
        result["plan_sha256"] = sha
        write_json(root / "supervision.json", result)
        if digest(args.plan) != sha:
            raise ValueError("diagnostic plan changed during run")
        report = summarize(args.plan, result)
        write_json(root / "aggregate.json", report)
        print({"completed": report["completed"], "aggregate_sha256": digest(root / "aggregate.json")},
              flush=True)
        if not report["completed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
