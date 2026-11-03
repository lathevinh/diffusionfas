"""Run the explicitly approved source-preparation batch, never training."""

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
    for name in ("metadata", "source-manifest", "fas-root", "detector", "private-root"):
        plan.add_argument(f"--{name}", type=Path, required=True)
    plan.add_argument("--config", type=Path, default=ROOT / "configs/offline-v1.json")
    plan.add_argument("--max-seconds", type=float, required=True)
    plan.add_argument("--max-rss-bytes", type=int, required=True)
    plan.add_argument("--max-output-bytes", type=int, required=True)
    plan.add_argument("--previous-plan", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--plan-sha256", help=argparse.SUPPRESS)
    report = commands.add_parser("report")
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "fas":
        raise ValueError("source preparation must use conda env fas")
    if arguments.command == "plan":
        from diffusionfas.source_preparation import build_plan

        path = build_plan(
            arguments.metadata, arguments.source_manifest, arguments.config, arguments.fas_root,
            arguments.detector, arguments.private_root, max_seconds=arguments.max_seconds,
            max_rss_bytes=arguments.max_rss_bytes, max_output_bytes=arguments.max_output_bytes,
            previous_plan=arguments.previous_plan,
        )
        print({"plan_sha256": digest(path), "completed": True}, flush=True)
        return
    if arguments.command == "report":
        from diffusionfas.source_preparation import summarize_preparation

        supervision = read_json(arguments.plan.parent / "supervision.json")
        if supervision["plan_sha256"] != digest(arguments.plan):
            raise ValueError("supervised plan commitment changed")
        report = summarize_preparation(arguments.plan, supervision)
        write_json(arguments.out, report)
        print({"completed": report["completed"], "aggregate_sha256": digest(arguments.out)}, flush=True)
        return
    if arguments.worker:
        from diffusionfas.source_preparation import run_preparation

        if arguments.plan_sha256 is None:
            raise ValueError("worker requires fixed parent plan commitment")
        run_preparation(arguments.plan, arguments.plan_sha256)
        return
    plan_sha = digest(arguments.plan)
    plan = read_json(arguments.plan, plan_sha)
    root = arguments.plan.parent
    if (root / "supervision.json").exists() or (root / "batch.log").exists():
        raise ValueError("batch already attempted; no automatic retry/resume")
    budget = plan["budget"]
    from diffusionfas.source_preparation import previous_output_root

    previous_root = previous_output_root(plan)
    result = run_bounded(
        [sys.executable, str(Path(__file__).resolve()), "run", "--plan", str(arguments.plan),
         "--worker", "--plan-sha256", plan_sha],
        cwd=ROOT, owned_root=root, log=root / "batch.log",
        max_seconds=budget["max_seconds"], max_rss_bytes=budget["max_rss_bytes"],
        max_disk_bytes=budget["max_output_bytes"],
        additional_output_roots=(previous_root,) if previous_root is not None else (),
    )
    result["plan_sha256"] = plan_sha
    write_json(root / "supervision.json", result)
    if digest(arguments.plan) != plan_sha:
        raise ValueError("supervised plan changed during batch")
    from diffusionfas.source_preparation import summarize_preparation

    report = summarize_preparation(arguments.plan, result)
    write_json(root / "aggregate.json", report)
    print({"completed": report["completed"], "prepared_originals": report["prepared_originals"],
           "aggregate_sha256": digest(root / "aggregate.json")}, flush=True)
    if not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
