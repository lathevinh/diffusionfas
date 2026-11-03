"""Bounded metadata-only source crop-fit audit. Never decode or score images."""

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
    plan.add_argument("--bonafide-manifest", type=Path, required=True)
    plan.add_argument("--diagnostic-root", type=Path, required=True)
    plan.add_argument("--private-root", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--plan-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "fas":
        raise ValueError("crop-fit audit requires conda env fas")
    from diffusionfas.crop_fit_audit import BUDGET, build_plan, run_audit
    if args.command == "plan":
        path = build_plan(args.bonafide_manifest, args.diagnostic_root, args.private_root)
        print({"plan_sha256": digest(path)}, flush=True)
    elif args.worker:
        if args.plan_sha256 is None:
            raise ValueError("audit worker needs parent plan commitment")
        run_audit(args.plan, args.plan_sha256)
    else:
        sha = digest(args.plan)
        root = args.plan.parent
        plan = read_json(args.plan, sha)
        if plan["budget"] != BUDGET:
            raise ValueError("fixed metadata audit caps required")
        if any((root / name).exists() for name in ("started.json", "supervision.json", "batch.log")):
            raise ValueError("audit already attempted; no automatic retry")
        result = run_bounded(
            [sys.executable, str(Path(__file__).resolve()), "run", "--plan", str(args.plan),
             "--worker", "--plan-sha256", sha],
            cwd=ROOT, owned_root=root, log=root / "batch.log", max_seconds=BUDGET["max_seconds"],
            max_rss_bytes=BUDGET["max_rss_bytes"], max_disk_bytes=BUDGET["max_output_bytes"])
        if digest(args.plan) != sha:
            raise ValueError("audit plan changed during execution")
        result["plan_sha256"] = sha
        write_json(root / "supervision.json", result)
        summary = read_json(root / "result.json") if (root / "result.json").exists() else None
        if summary is not None and summary["plan_sha256"] != sha:
            raise ValueError("audit result plan mismatch")
        report = {"completed": bool(result["completed"] and summary is not None),
                  "supervision": result, "result": summary}
        write_json(root / "aggregate.json", report)
        print(report, flush=True)
        if not report["completed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
